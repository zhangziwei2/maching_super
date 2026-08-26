# 机床故障诊断 RAG 存储规则与知识图谱抽象设计规范

> 适用范围：`maching` 项目（机床故障诊断）
> 借鉴对象：Yuxi 项目（`D:\Azhangziwei\AgentChef312\Yuxi`）的知识图谱设计
> 定位：本规范定义**向量库（Milvus）存储规则**与**知识图谱（Neo4j）故障原因实体关系抽象层**，
> 以及二者通过 **mix 混合检索**融合的接入方式。

> ⚠️ **本文为设计草案 / 目标架构**（Neo4j + LightRAG 双轨 + `fault_domain` 标签隔离）。
> 当前（2026-08-19）实际部署：领域规则图谱运行于 **NetworkX**（`kg/`，无外部数据库，详见 `KNOWLEDGE_GRAPH_UPGRADE.md`），
> Phase 2a 已新增 `backend/graphkb/` 统一图谱层，将「手工三元组」与「LightRAG 自动图谱」接入 **Neo4j**，
> 并通过 `GRAPH_FUSION` / `GRAPH_SOURCES` 程度开关与领域图谱融合（详见 `KNOWLEDGE_GRAPH_UPGRADE.md` 第 14 节）。

---

## 1. 目标与定位

maching 当前有两套能力但彼此割裂：

- **RAG 侧（成熟）**：`backend/milvus_client.py` 已实现稠密+稀疏混合检索、三级 Auto-merging、RRF 融合；
  `backend/rag_pipeline.py` 是 LangGraph，含 grade / rewrite（step_back / hyde / complex）/ expand。
- **KG 侧（偏初级）**：`kg/build_machinegraph_full.py` 手建了 `FaultType–Cause` 等因果图谱；
  `kg/graphrag_enhancer.py` 目前只是**关键词匹配占位**，没有真正接入 RAG 检索。

本规范要解决的核心问题：**让知识图谱作为"故障原因的实体关系抽象层"正式进入 RAG 检索**，
弥补向量检索"只给片段、不给因果"的短板。具体对标 Yuxi 的四条经验：

| Yuxi 机制 | 在 maching 的落地 |
|-----------|------------------|
| 双轨图谱（LightRAG 自动轨 + 手动三元组轨） | 轨A：专家审核的 `machine_fault.json` 手建；轨B：新维修手册/案例用 LightRAG 自动抽取 |
| `kb_id` 标签隔离 | `fault_domain` 标签隔离（车削/铣削/五轴等不同工艺域） |
| `GraphAdapter` 统一接口 | `backend/kg_adapter.py` 统一两轨查询 |
| `mode=mix` 混合检索 | RAG 管线新增 KG 检索节点，向量轨 + 图轨融合 |

---

## 2. 总体架构

```
数据源               构建轨                     存储层                      检索层               消费
─────────────────────────────────────────────────────────────────────────────────────────────
machine_fault.json ─┐                                                      ┌─ 向量轨(Milvus) ─┐
（专家/权威）        ├─ 轨A 精选手建 ──► Neo4j(图谱) ─┐                    │                  │
                    │                 (FaultType…)   ├─ fault_domain ──►  ├─ mix 融合 ───────► LLM
维修手册PDF/案例 ───┘                                  │   标签隔离       │                  │
                    └─ 轨B LightRAG ─► Neo4j+Milvus ─┘                    └─ 图轨(KG抽象) ──┘
                      (自动抽取实体关系)
```

- **向量库**：负责"语义片段召回"（已有，Milvus 混合检索 + Auto-merging）。
- **图谱库**：负责"故障原因实体关系抽象"（新增/补强，Neo4j）。
- 两库共享同一 `fault_domain` 命名空间做隔离，互不影响。

---

## 3. 存储分层与职责

| 层 | 引擎 | 存什么 | 对应 Yuxi |
|----|------|--------|-----------|
| 向量层 | Milvus | 文档分块 + dense/sparse 向量 + 元数据 | Milvus 知识库 |
| 图层层 | Neo4j | 故障–原因–机理–部位–措施的实体与关系 | LightRAG 图谱 + 全局图谱 |
| 元数据/KV 层 | 文件系统/SQLite | 文档状态、content_hash 去重、分块索引 | JsonKVStorage / doc_status |

**关键原则**：图谱**不是把文档再存一遍**，而是抽取出"故障因果骨架"。
同一份文档既进向量库（片段），也进图谱库（因果），但职责不同。

---

## 4. 向量库（Milvus）存储规则

> 本节与现有 `backend/milvus_client.py` 保持一致，作为团队统一约定。

### 4.1 分块规则（三级分块 + Auto-merging）

- 采用**三级分块**：叶子块 `L3` → 父块 `L2` → 根块 `L1`。
- 每个块必须带层级字段，便于召回后向上合并：

| 字段 | 含义 |
|------|------|
| `chunk_id` | 当前块唯一 ID |
| `parent_chunk_id` | 父块 ID |
| `root_chunk_id` | 根块 ID |
| `chunk_level` | 层级（1/2/3） |
| `chunk_idx` | 块在文档内的序号 |

- **检索规则**：从叶子块（`chunk_level=3`）召回，命中后若有相邻命中则 Auto-merge 到父块，避免语义割裂。
- 已有实现：`MilvusManager.hybrid_retrieve` + `get_chunks_by_ids`（拉父块）。

### 4.2 嵌入规则

| 类型 | 模型 | 维度 | 索引 | 度量 |
|------|------|------|------|------|
| dense | BAAI/bge-m3 | 1024（`DENSE_EMBEDDING_DIM`） | HNSW | IP |
| sparse | BM25 | — | SPARSE_INVERTED_INDEX | IP |

- 混合检索使用 **RRF 融合**（`RRFRanker(k=60)`），两路各取 `top_k*2` 再融合。
- 降级：稀疏不可用时退回纯 dense（`dense_retrieve`）。

### 4.3 元数据字段规范（写入 Milvus 的每行必须包含）

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | VARCHAR(2000) | 分块文本 |
| `filename` | VARCHAR(255) | 来源文件名 |
| `file_type` | VARCHAR(50) | 文档类型 |
| `file_path` | VARCHAR(1024) | 原始路径 |
| `page_number` | INT64 | 页码 |
| `chunk_idx` | INT64 | 块序号 |
| `chunk_id` / `parent_chunk_id` / `root_chunk_id` | VARCHAR(512) | 层级链 |
| `chunk_level` | INT64 | 层级 |
| `fault_domain` | VARCHAR(64) | **工艺域标签**（新增，用于隔离） |
| `content_hash` | VARCHAR(64) | **内容哈希**（新增，去重用） |

> 新增的 `fault_domain` / `content_hash` 是为了对齐 Yuxi 的 `kb_id` 隔离与 content-hash 去重。

### 4.4 集合 Schema（对齐 `milvus_client.py.init_collection`）

主键 `id`(INT64, auto_id)；向量 `dense_embedding`(FLOAT_VECTOR,1024) + `sparse_embedding`(SPARSE_FLOAT_VECTOR)；
文本/元数据见上表；层级字段 `chunk_id/parent_chunk_id/root_chunk_id/chunk_level`。
索引：dense→HNSW(M=16, efConstruction=256)，sparse→SPARSE_INVERTED_INDEX(drop_ratio_build=0.2)。

### 4.5 检索规则

- 入口 `top_k=5`；模式 `hybrid`（dense+sparse）+ rerank（可选）。
- 查询扩展（LangGraph 节点 `rewrite_question_node`）：
  - `step_back`：含具体名称/参数，先抽象到通用概念；
  - `hyde`：模糊概念性问题，生成假设性文档再检索；
  - `complex`：多步问题，两者叠加。
- 相关性门控：`grade_documents_node` 用 grader 模型二分类，不相关则走扩展检索。

### 4.6 写入 / 更新 / 删除

- **去重**：入库前计算 `content_hash`，相同哈希跳过（对标 Yuxi 基于 content_hash 去重）。
- **更新**：删除 `chunk_id` 同文档的旧块后重写（参考 Yuxi `delete_file_chunks_only`）。
- **删除**：按 `file_path` 或 `content_hash` 过滤表达式删除；图谱侧按 `fault_domain` 标签 `DETACH DELETE`。

---

## 5. 知识图谱（Neo4j）抽象规范 —— 故障原因实体关系抽象

### 5.1 设计原则

KG 对外暴露的不是"数据库"，而是**故障因果的抽象骨架**：

```
故障(FaultType) ──HAS_CAUSE──► 原因(Cause) ──CAUSED_BY_MECHANISM──► 机理(Mechanism)
      │                              │
   HAS_SYMPTOM                    INVOLVES_COMPONENT
      ▼                              ▼
  现象(Phenomenon)              部件(Component)
      │
   HAS_SOLUTION / NEEDS_CHECK / HAS_PREVENT / HAS_PARAMETER
```

LLM 拿到这份抽象后，把它当作"推理骨架"：先确定故障节点 → 沿 `HAS_CAUSE` 找到根因 →
沿 `HAS_SOLUTION` 给措施。答案可溯源到具体故障节点，可解释、可审计。

### 5.2 实体类型（聚焦故障原因）

| 标签 | 含义 | 是否已有 |
|------|------|----------|
| `FaultType` | 故障类型（中心节点） | ✅ |
| `Cause` | 故障直接原因 | ✅ |
| `Mechanism` | 故障机理（原因背后的物理/工艺机制，建议新增） | ➕ |
| `Component` | 机床部件 | ✅ |
| `Phenomenon` | 故障现象/症状 | ✅ |
| `Solution` | 解决方法 | ✅ |
| `Check` | 检测方法 | ✅ |
| `Parameter` | 加工参数建议 | ✅ |
| `Material` | 适用材料 | ✅ |
| `Prevent` | 预防措施 | ✅ |
| `CureWay` | 修复方式 | ✅ |
| `Category` | 故障分类 | ✅ |

> 建议新增 `Mechanism`：把"为什么会产生这个 Cause"显式建模，让原因抽象更深入一层。

### 5.3 关系类型

| 关系 | 方向 | 语义 |
|------|------|------|
| `HAS_CAUSE` | FaultType → Cause | 故障的直接原因 |
| `CAUSED_BY_MECHANISM` | Cause → Mechanism | 原因背后的机理 |
| `INVOLVES_COMPONENT` | FaultType → Component | 涉及部件 |
| `HAS_SYMPTOM` | FaultType → Phenomenon | 表现现象 |
| `HAS_SOLUTION` | FaultType → Solution | 解决措施 |
| `NEEDS_CHECK` | FaultType → Check | 检测手段 |
| `HAS_PARAMETER` | FaultType → Parameter | 参数建议 |
| `APPLIES_TO_MATERIAL` | FaultType → Material | 适用材料 |
| `HAS_PREVENT` | FaultType → Prevent | 预防 |
| `HAS_CURE_WAY` | FaultType → CureWay | 修复方式 |
| `BELONGS_TO_CATEGORY` | FaultType → Category | 归类 |

### 5.4 双轨构建（借鉴 Yuxi 双轨）

- **轨A·精选权威轨（手建确定性）**
  - 数据源：`kg/data/machine_fault.json`（专家审核，高精度）。
  - 构建：`kg/build_machinegraph_full.py`（py2neo 确定性写入）。
  - 角色：**故障原因抽象的主来源**，可信、可控、可解释。
- **轨B·自动扩展轨（LightRAG）**
  - 数据源：新到的维修手册 PDF、历史维修案例、FMEA 文档。
  - 构建：参考 Yuxi `backend/package/yuxi/knowledge/implementations/lightrag.py`，
    用 `LightRAG.ainsert` 自动抽取实体关系，落 Neo4j(`Neo4JStorage`) + Milvus(`MilvusVectorDBStorage`)。
  - 角色：覆盖轨A 未收录的长尾故障与案例，自动补全。
- 两轨**共享同一 Neo4j 实例**，通过 `fault_domain` 标签隔离（见 5.5）。

### 5.5 隔离规则（借鉴 Yuxi 的 `kb_id`）

- 每个工艺域一个 `fault_domain` 标签，例如：`LATHE`（车削）、`MILL_3AXIS`（三轴铣）、`MILL_5AXIS`（五轴）。
- 节点写入时统一带 `fault_domain` 标签与 `name` 属性（对齐 Yuxi 注意事项：每个节点都要 `Entity` 标签 + `name`）。
- Cypher 查询必须带 `fault_domain` 过滤，**禁止跨域**召回：

```cypher
MATCH (f:FaultType:`LATHE` {name:'主轴过热'})
OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
RETURN f.name AS fault, collect(c.name) AS causes
```

- 删除：按 `fault_domain` 标签 `MATCH (n:`<domain>`) DETACH DELETE n`（对齐 Yuxi `delete_database`）。

### 5.6 KG 抽象层接口（GraphAdapter，借鉴 Yuxi 的 `graphs/adapters/lightrag.py`）

新增 `backend/kg_adapter.py`，统一轨A/轨B 的查询出口，对上层只暴露：

| 方法 | 作用 |
|------|------|
| `query_cause_abstraction(query, domain) -> subgraph` | 返回"故障→原因→机理"因果子图（文本化或结构） |
| `expand_neighbors(node, hops=1)` | 1-hop 邻居展开（现象/部件/措施） |
| `get_stats(domain)` | 节点/关系统计（用于可视化与校验） |
| `normalize_node / normalize_edge` | 统一输出结构，抹平轨A(py2neo) 与轨B(LightRAG) 差异 |

这样上层 RAG 管线**不关心**数据来自手建还是 LightRAG，只调用 `query_cause_abstraction`。

### 5.7 为什么 KG 提供的是"原因实体关系抽象"而不是"另一份文档"

| 维度 | 向量检索（RAG 片段） | 知识图谱（原因抽象） |
|------|----------------------|----------------------|
| 返回 | 一段相似文本 | 故障→原因→机理的结构化链 |
| 因果完整性 | 取决于片段是否凑巧包含 | 显式建模，必然完整 |
| 可解释性 | 低（黑盒相似度） | 高（可溯源到节点） |
| 多跳推理 | 弱 | 强（沿关系走多跳） |
| 示例 | "主轴过热可能因为轴承…" | `主轴过热─HAS_CAUSE→轴承润滑不良─CAUSED_BY_MECHANISM→摩擦热积聚` |

---

## 6. KG 与 RAG 融合规则（mix 模式，借鉴 Yuxi）

### 6.1 检索流程（接入 `backend/rag_pipeline.py`）

```
用户问题
   │
   ├─► 向量轨：retrieve_initial(Milvus hybrid) ──► top_k=5 chunks
   │
   └─► 图轨：kg_retrieve(KGAdapter.query_cause_abstraction) ──► cause-subgraph
                          │
                          ▼
                  mix 融合 context = chunks + subgraph文本化
                          │
                          ▼
                 grade / rewrite / expand（现有 LangGraph）
                          │
                          ▼
                        LLM 生成
```

### 6.2 融合策略

- 向量轨召回 `top_k=5` 片段，图轨召回"故障→原因"因果子图（实体+关系）。
- **context = chunks + KG subgraph 文本化**。当两个来源命中同一故障，KG 因果链作为"结论骨架"，
  chunks 作为"证据细节"。
- 若图轨无命中（如全新故障），降级为纯向量检索，不影响主链路。

### 6.3 接入点（改造清单）

| 文件 | 改造 |
|------|------|
| `backend/kg_adapter.py` | **新增**：GraphAdapter 风格统一接口 |
| `backend/rag_pipeline.py` | **新增** `kg_retrieve` 节点，在 `retrieve_initial` 之后并行调用，结果并入 `context` |
| `backend/kg_tool.py` | 把 `@tool("query_knowledge_graph")` 内部改为调用 `KGAdapter`（目前直接调 `chatbot_graph`） |
| `kg/graphrag_enhancer.py` | 由"关键词匹配"升级为"向量语义检索"（接 Milvus），作为 mix 的辅助增强 |
| `kg/build_machinegraph_full.py` | 写入时补 `fault_domain` 标签与 `name` 属性 |

---

## 7. 配置与隔离（`.env`）

```env
# 向量库
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=embeddings_collection
DENSE_EMBEDDING_DIM=1024

# 图库
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=****

# 工艺域列表（fault_domain 隔离）
FAULT_DOMAINS=LATHE,MILL_3AXIS,MILL_5AXIS

# LightRAG 自动轨（轨B，可选）
EMBEDDING_TIMEOUT=60
LLM_TIMEOUT=180
```

---

## 8. 落地步骤（实现路线）

1. **固化轨A**：给 `machine_fault.json` 的导入脚本补 `fault_domain` 标签写入，建立隔离基线。
2. **引入 LightRAG（轨B）**：新增 `backend/knowledge/lightrag_impl.py`，参考 Yuxi 的
   `LightRagKB._create_kb_instance`（Neo4JStorage + MilvusVectorDBStorage + ragflow_like 分块）。
3. **抽象层**：新增 `backend/kg_adapter.py`，实现 5.6 的四个方法，抹平两轨差异。
4. **融合节点**：在 `rag_pipeline.py` 加 `kg_retrieve`，mix 融合进 `context`。
5. **升级增强器**：把 `graphrag_enhancer.py` 从关键词改为接 Milvus 的向量语义检索。
6. **评测**：用固定故障问答集对比"纯向量" vs "mix"，验证 KG 抽象带来的可解释性与准确率提升。

---

## 9. 与现状的差异（改造点清单，速查）

| 模块 | 现状 | 目标 |
|------|------|------|
| `kg/graphrag_enhancer.py` | 关键词匹配占位 | 向量语义检索（接 Milvus） |
| `backend/rag_pipeline.py` | 仅向量检索 | 新增 `kg_retrieve` 图轨节点 |
| `backend/kg_tool.py` | 直接调 `chatbot_graph` | 改为调 `KGAdapter` |
| `kg/build_machinegraph_full.py` | 无 domain 隔离 | 补 `fault_domain` 标签 |
| 存储元数据 | 无 `fault_domain`/`content_hash` | Milvus 行增加两字段 |
| 自动抽取 | 无 | LightRAG 轨B 自动补图 |

---

> 本规范是 maching 项目的**存储与抽象约定文档**。实施时先落第 1、3、4 步（隔离基线 + 抽象层 + 融合节点），
> 即可让 KG 作为"故障原因实体关系抽象"进入 RAG；轨B(LightRAG) 可作为第二阶段增量扩展。
