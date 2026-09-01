# Maching — 机床故障诊断知识图谱 + RAG 智能问答系统

面向**机床智能诊断**工业场景的检索增强问答系统。以「知识图谱 + 文档 RAG」**全量双路召回**为核心，配合查询改写、结果重排、三级分块 Auto-merging、**可回答性门控**与**依据来源标注**，在保证答案可溯源的前提下降低幻觉；工程侧通过模型服务解耦、任务队列、上下文压缩与记忆系统，支撑多用户并发、长时间入库与跨会话记忆。系统同时兼容医疗、教育等行业知识库场景。

---

## 一、核心功能

### 1. 双通道全量召回（KG + RAG 恒定并发）
每一轮提问都**并发双查**两个知识源，不再由意图路由决定"跳过哪一路"（路由仅保留用于可观测性），彻底消除漏召回：

- **知识图谱通道（KG）**：三源统一融合（`backend/graphkb`，按 `GRAPH_SOURCES` / `GRAPH_FUSION` 开关控制）：
  - **领域规则图谱**：内置 NetworkX 图引擎 + AC 自动机实体锚定 + 1~2 跳 BFS 子图扩展，输出结构化三元组与故障属性（类型/原因/处理建议），确定性强、可解释；
  - **手工三元组图谱**：Neo4j `Entity:Upload`，节点带 bge-m3 向量，向量（余弦）+ 模糊双路检索；
  - **LightRAG 自动图谱**：上传文档自动抽取实体关系，落 Neo4j。
- **文档 RAG 通道（Milvus）**：
  - **混合检索**：dense 向量（bge-m3，1024 维，归一化内积=余弦）+ Milvus 原生 BM25 稀疏向量，RRF（k=60）融合，兼顾语义匹配与关键词精确命中；
  - **Cross-Encoder 重排**：`bge-reranker-v2-m3` 对候选片段精排（候选池 = top_k×3=15 → top5）；
  - **Auto-merging 三级检索**：三级嵌套分块（L1 1200 / L2 600 / L3 300），只检索叶子层（L3）再按"同一父块命中 ≥2"自动上卷合并至 L2/L1，兼顾检索精度与上下文完整度；
  - **LangGraph 流水线**：检索 → LLM 相关性打分（grade）→ 不相关时查询改写（step_back / hyde / complex）→ 扩展检索。

### 2. 可回答性门控（抗幻觉，关键特色）
生成答案前对检索结果做**纯规则、零额外 LLM** 的判定，依据不足直接短路拒答：
- **两源皆空** → 硬拒答：明确告知知识库无相关内容，不臆造；
- **文档侧四态**（`answerability.evaluate`）：`hard_reject` / `soft_reject`（Top-1 重排分 < 0.3，可调）/ `conflict`（共享主题词 + 结论词冲突检测）/ `pass`；
- **仅图谱命中** → 放行，但注入【依据提示】要求模型"文档未覆盖时不要凭通用知识补具体条款"（防安全规程编造）。

门控注册为 **critical Hook**，失效以异常暴露而非静默跳过；判定结果透出至 `rag_trace` 便于审计。

### 3. 依据来源标注
`_build_context` 为两路结果分别标注 `【知识图谱检索结果】(三元组，用于故障因果推理)` 与 `【文档知识库检索结果】(来自上传文档原文，优先采用)`，让 LLM 区分"图谱因果"与"手册原文"，避免把三元组当操作规程使用。

### 4. 上下文压缩 + 记忆系统
- **上下文压缩**（`context_compact.py`）：L1 超长检索片段落盘为 blob（预览 + 指针）；历史消息按字符预算分级压缩（L2/L3/L4）；被压缩的原始对话归档 transcript，可追溯、不销毁。
- **记忆系统**（`memory.py`）：跨会话记忆，多用户隔离 + 机器级公共记忆（`scope=machine`，仅管理员可写）；`mem_type` 区分用户画像/纠正反馈/设备事实/外部引用；Stop 钩子显式触发抽取、PreGenerate 注入上下文。

### 5. Hooks 运行时（横切能力收编）
将门控、检索进度、上下文压缩、记忆抽取等收编为统一钩子（`UserPromptSubmit / PreRetrieve / PostRetrieve / PreGenerate / PostGenerate / Stop`），新增横切能力零改动主流程。

### 6. 文档管理与异步入库
- 支持 PDF / Word（.docx/.doc）/ TXT / Markdown / Excel 上传，自动完成解析、三级分块、父块（L1/L2）入库 PG + Redis、叶子（L3）向量化入库 Milvus，并可选自动抽取实体关系图谱；
- **同名去重**：重传同名文件自动清理旧版本（Milvus 向量 + BM25 统计 + PG 父块三处同步删除后重写）；
- 上传/删除任务由 **Redis Stream 任务队列**调度，失败自动重试（最多 3 次，指数退避），Redis 不可用时自动降级线程池，消费者线程自愈；
- 任务状态持久化于数据库，服务重启不丢失，前端轮询展示分步进度。

### 7. 其他能力
- **用户与会话体系**：JWT 认证、角色区分（管理员管理文档/图谱）、多会话历史持久化 + 归档。
- **颤振诊断**：基于振动信号多层融合（SAE + Stacking）的颤振自动诊断模块，支持 CSV/XLSX 分析、多模式投票与实时监测报警。
- **评测集**：`eval_golden.json`（30 题）+ `eval_report.json` 量化 rerank / 门控效果。

---

## 二、环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.12 | 项目声明 `requires-python = ">=3.12"` |
| Docker / Docker Compose | 最新稳定版 | 用于启动依赖中间件 |
| PostgreSQL | 15（容器提供） | 父块存储、会话、记忆与任务状态 |
| Redis | 7（容器提供） | 缓存（父块 / 会话 / 记忆）与任务队列（Stream） |
| Milvus | 2.5.x（容器提供） | 混合检索向量库，需支持原生 BM25 Function |
| Neo4j | 5（容器提供，可选） | 手工三元组图谱 + LightRAG 自动图谱（`GRAPH_SOURCES` 未启用时可省） |
| GPU（推荐） | CUDA 可选 | 加速 embedding / rerank 推理；无 GPU 可设 `EMBEDDING_DEVICE=cpu`、`RERANK_DEVICE=cpu` |

> 注意：`pymilvus` 需 **≥ 2.5**（原生 BM25 依赖）。若使用旧版，`hybrid_retrieve` 会给出明确报错提示。

---

## 三、安装与启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

国内网络建议先配置 HuggingFace 镜像，避免模型下载失败：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 启动依赖中间件

```bash
docker compose up -d
```

该命令启动 PostgreSQL、Redis、Milvus（含 etcd / MinIO）、Attu 管理界面与 Neo4j。等待 Milvus 健康检查通过后再进行下一步。

### 3. 配置环境变量

在项目根目录创建 `.env`，关键项如下（其余均有默认值，见 `docs/TECH_GUIDE.md`）：

```env
# ===== LLM（OpenAI 兼容）=====
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# ===== 模型推理服务 =====
EMBEDDING_SERVICE_URL=http://127.0.0.1:8002   # embedding 服务
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda                          # 无 GPU 用 cpu
RERANK_BINDING_HOST=http://127.0.0.1:8001      # rerank 服务
RERANK_MODEL=BAAI/bge-reranker-v2-m3

# ===== 存储 =====
MILVUS_HOST=localhost
MILVUS_PORT=19530
DENSE_EMBEDDING_DIM=1024
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app
REDIS_URL=redis://localhost:6379/0

# ===== 检索与门控（均有默认值，可调）=====
AUTO_MERGE_ENABLED=true
AUTO_MERGE_THRESHOLD=2
LEAF_RETRIEVE_LEVEL=3
ANSWERABILITY_RERANK_THRESHOLD=0.3

# ===== 知识图谱（可选）=====
GRAPH_SOURCES=domain,manual_triples,lightrag
GRAPH_FUSION=hybrid
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=****

# ===== 上传 =====
MAX_UPLOAD_MB=200
```

### 4. 启动模型推理服务（两个独立进程）

模型推理已从 API 进程解耦，需**单独启动**以下两个服务：

```bash
# 终端 1：Embedding 服务（端口 8002，bge-m3）
python -m backend.embedding_server

# 终端 2：Rerank 服务（端口 8001，bge-reranker-v2-m3）
python tests/rerank.py
```

服务就绪后可通过 `http://127.0.0.1:8002/health` 验证 embedding 服务状态。

> 模型仅在独立进程中加载一次，API 进程纯 HTTP 调用；首次启动会自动下载模型（bge-m3 约 2GB，reranker 约 568MB）。

### 5. 启动主服务

```bash
python main.py
```

启动后自动打开 `http://localhost:8000`；也可使用 `python -m backend.app` 启动（不自动开浏览器）。

### 6.（首次部署 / 升级必读）重建向量集合

由旧版本升级、或首次初始化集合时，需执行全量重灌以应用新的 BM25 schema：

```bash
python -m backend.rebuild_milvus --yes
```

该脚本从 PostgreSQL 的 `parent_chunks` 表全量读取分块，重建 Milvus 集合后重新向量化写入。**属破坏性操作**，会清空现有向量集合（PG 数据不受影响），默认需交互确认，`--yes` 可跳过确认。

> ⚠️ 注意：`parent_chunks` 表在正常上传流程中只落 L1/L2 父块（叶子 L3 只进 Milvus）。重建前请确认表中是否包含叶子层数据，否则重建后 `chunk_level==3` 的叶子检索会查空。
>
> 旧版本使用自实现 BM25（本地 JSON 状态），与新 schema 不兼容，必须重建后才能正常检索。

### 7. 训练颤振诊断模型【可选】

```bash
python -m backend.chatter.train_all
```

---

## 四、第二轮迭代更新

本轮迭代围绕**「抗幻觉、降延迟、提可用性」**三条主线，共完成 6 项新增与升级。

### 新增功能
1. **Milvus 全量重灌脚本（`backend/rebuild_milvus.py`）**
   新增集合重建脚本，从 PostgreSQL 全量恢复分块数据，支持 `--yes` 非交互执行，用于 schema 升级后的数据迁移。

2. **Embedding 独立推理服务（`backend/embedding_server.py`）**
   新增端口 8002 的 HTTP 推理服务（`/v1/embeddings`，OpenAI 兼容格式），支持可选 API Key 鉴权与 `/health` 健康检查。

3. **Redis Stream 任务队列（`backend/task_queue.py`）**
   新增基于 `XADD` / `XREADGROUP` / `XACK` 的任务调度，支持失败自动重试 3 次（指数退避 1s/2s/4s），并在 Redis 不可用时自动降级线程池，保证入库任务不丢失、不阻塞。

### 功能升级

4. **稀疏检索迁移至 Milvus 原生 BM25**
   移除自实现的 BM25（本地 JSON 词表与词频状态），改为在 Milvus schema 中注册原生 `FunctionType.BM25`（text 字段启用中文 analyzer），稀疏索引切换为 `metric_type="BM25"`。
   **收益**：消除多进程间词表不一致与状态文件损坏风险，IDF 统计由服务端自动维护，删除数据后仅需 `flush` 即可刷新，彻底解决旧方案"单文件状态"的可靠性瓶颈。

5. **Embedding 服务解耦与主接口全异步化**
   - `backend/embedding.py` 精简为纯 HTTP 客户端，API 进程不再加载模型、无本地兜底，模型只驻留独立进程，支持独立扩容与重启；
   - 主聊天接口 `chat_endpoint` 改用 `asyncio.to_thread`，流式链路中的同步 LLM 调用移入线程池，避免阻塞 FastAPI 事件循环；
   - 意图路由重写为纯规则实现（去除原有模型依赖），消除冗余开销。

### 兼容性说明

- 依赖变更：`pymilvus` 需升级至 **≥ 2.5**；
- 启动变更：需额外启动 Embedding 服务（8002）与 Rerank 服务（8001）；
- 数据变更：**必须执行** `python -m backend.rebuild_milvus --yes` 重建向量集合，否则旧数据无法与原生 BM25 索引配合检索。

---

## 五、第三轮迭代更新

本轮迭代以**「安全加固、Bug 修复、简化瘦身」**为主线，聚焦代码质量与运行稳定性，共完成 9 项修复。

### 安全加固

1. **修复图片上传路径穿越漏洞**
   `/upload/image` 原直接使用客户端传入的 `file.filename` 拼接落盘路径，攻击者可传入 `../../` 覆盖项目任意文件（如 `.env`、源码）。
   新增 `_safe_filename()` 统一净化文件名（取 `Path().name` 基名 + 剔除 `<>:"|?*` 与控制字符），并叠加 uuid 前缀。同类隐患 `/kg/triples/import` 一并修复。

2. **上传大小限制**
   新增统一落盘函数 `_save_upload_stream()`，流式分块写入并实时累计字节数，**超限立即中断并清理半截文件**（返回 413）。默认上限 200MB，可用 `MAX_UPLOAD_MB` 环境变量调整，杜绝大文件撑爆磁盘。

3. **权限收紧**
   `/kg/triples/import` 由「任意登录用户」改为 `require_admin`。该端点会将三元组文件持久落盘至 `kg/data/triples/` 并污染知识库，重建图谱后仍保留，不应开放给普通用户。

4. **删除硬编码凭据**
   删除废弃脚本 `kg/simple_chatbot.py`（内含明文数据库密码）。该脚本属已废弃的 Neo4j 版本，无任何引用。

### Bug 修复

5. **修复并发请求间状态串号（关键缺陷）**
   `tools.py` 中检索上下文、工具调用计数、RAG 步骤队列原为**模块级全局变量**，而检索在线程池中并发执行，导致多用户同时提问时：A 的检索上下文被 B 覆盖、进度步骤串到 B 的输出流、工具计数互相干扰。
   - 将 4 个全局变量改为 `contextvars.ContextVar`（对 asyncio 任务天然隔离）；
   - 检索调用改为模块级复用线程池 + `contextvars.copy_context()` 显式传递上下文快照，双保险保证请求间彻底隔离；
   - 附带消除原「每次请求新建线程池」的开销。

6. **异常静默改为可观测日志**
   `_search_kg`、`_search_rag`、`search_knowledge_base` 三处异常原一律返回空串且不记录，导致 Milvus 不可达、图谱加载失败时仅表现为「答非所问」，极难排查。现统一补 `logger.warning(..., exc_info=True)` 留痕。

7. **文档列表分页拉取**
   `/documents` 原一次性拉取 10000 条分块在内存聚合，文档量增长后会撞上 Milvus 单次查询窗口上限（16384）。改用 `query_all()` 按窗口分页拉取，彻底规避上限风险。

### 简化瘦身
8. **统一文件上传逻辑**
   原 4 处重复的 1MB 分块写盘循环（`_save_chatter_upload`、`_save_upload_file`、`/upload/image` 内联、`/kg/triples/import` 内联）合并为单个 `_save_upload_stream()`，消除重复代码，并统一获得路径净化与大小限制能力。

### 兼容性说明

- **行为变更**：`/upload/image` 返回的 `filename` 字段现包含 uuid 前缀（如 `a1b2c3d4_photo.png`），如需展示原名请自行截取或告知调整；
- **新增环境变量**：`MAX_UPLOAD_MB`（上传大小上限，默认 200）；
- **数据/启动无变更**：本轮为纯代码层修复，**无需**重建向量集合或改动启动流程。

---

## 六、第四轮迭代更新（KG 升级 + 全量双路召回 + 门控升级）

本轮以**「KG 三源融合、召回准确性、抗幻觉门控」**为主线，共完成 6 项升级。

### 新增功能

1. **KG 三源统一图谱层（`backend/graphkb/`）**
   - `GraphKB` 统一服务汇聚三类来源，按 `GRAPH_SOURCES` / `GRAPH_FUSION` / `GRAPH_WEIGHT` 开关融合，单源故障优雅降级：
     - `domain`：领域规则图谱（成熟 NetworkX `kg.graph_service`）；
     - `manual_triples`：手工/专家三元组 JSONL → Neo4j `Entity:Upload`（`upload_graph.py`，节点 bge-m3 向量 + `entityEmbeddings` 余弦向量索引 + 子串模糊兜底）；
     - `lightrag`：文档自动抽取图谱（`lightrag_kb.py`，LightRAG 1.x：图落 Neo4j `Neo4JStorage`、向量 `NanoVectorDBStorage`）。
   - `query_knowledge_graph` 优先走统一层，失败回退 legacy NetworkX（`tools.py`）。

2. **全量双路召回（取代「路由跳源」）**
   - 意图路由**不再决定跳过哪一路**：KG 与 RAG 恒定并发双查（线程池 + 各自 `contextvars` 快照），路由结果仅写入 `rag_trace` 用于可观测性。
   - 修复漏召回根因：问「主轴安全说明 2.1 注意事项」原被路由判为纯 KG 问题、文档库从未被访问，导致模型凭通用知识编造安全条款。

3. **依据来源标注**
   - `_build_context` 为两路结果标注来源，并在「仅图谱命中」时注入【依据提示】，要求模型"文档未覆盖时不要凭通用知识补充具体条款或操作步骤"。

4. **可回答性门控升级**
   - `answerability_gate` 改为按**合计依据**判定（两源皆空硬拒 / 文档有内容走四态 / 仅图谱放行但标注依据），注册为 critical Hook，失效以异常暴露；
   - `answerability.evaluate` 完善四态判定：`hard_reject` / `soft_reject`（Top-1 rerank 分 < 0.3，`ANSWERABILITY_RERANK_THRESHOLD` 可调）/ `conflict`（共享主题词 + 否定/肯定结论词冲突检测）/ `pass`，全部零额外 LLM 调用。

5. **评测与回归验证**
   - `scripts/verify_dual_retrieval.py`：路由判定、依据标注、门控判定三组纯函数回归 + `--live` 端到端；
   - `eval_golden.json`（30 题）+ `eval_report.json`：rerank 应用 30/30，Top-1 软拒答阈值可评估。

### 兼容性说明

- 新增环境变量：`GRAPH_SOURCES` / `GRAPH_FUSION` / `GRAPH_WEIGHT` / `NEO4J_*` / `ANSWERABILITY_RERANK_THRESHOLD`；
- 图谱层为渐进升级，未启用 `graphkb` 来源时行为与上一版一致。

---

## 七、第五轮迭代更新（记忆 / 上下文压缩 / Hooks / 健壮性）

本轮以**「长会话能力、横切解耦、上传健壮性」**为主线，聚焦运行时能力与稳定性。

### 新增功能

1. **Hooks 运行时（`backend/hooks.py` + `hooks_builtin.py`）**
   将门控、检索完成进度、上下文压缩、记忆抽取等横切逻辑收编为统一钩子：`UserPromptSubmit / PreRetrieve / PostRetrieve / PreGenerate / PostGenerate / Stop`。主流程不再包含策略判断，新增横切能力仅需注册钩子，零改动主流程。

2. **上下文压缩（`backend/context_compact.py`）**
   - **L1 上下文预算**：超长检索片段落盘为 blob，替换为「预览 + blob_id 指针」，把单次 prompt 约束在预算内（Auto-merging 上卷后上下文可能无界膨胀，此为第一道闸门）；
   - **历史分级压缩（L2/L3/L4）**：历史消息按字符预算逐级摘要；
   - **可追溯归档**：被压缩的原始对话由 `on_archive` 回调归档 `chat_transcripts`，压缩从"不可逆销毁"变为"移出上下文但可追溯"。

3. **记忆系统（`backend/memory.py`）**
   - 跨会话记忆，多用户隔离（`user_id`）+ 机器级公共记忆（`scope=machine`，仅管理员可写，防止学员对话污染公共设备档案）；
   - `mem_type` 区分 `user`（画像）/ `feedback`（纠正反馈）/ `project`（设备事实）/ `reference`（外部引用）；
   - Stop 钩子显式信号触发 LLM 抽取（并发不串号）、PreGenerate 记忆注入上下文。

4. **上传与任务健壮性**
   - 统一 `_save_upload_stream`：路径净化 + 大小限制 + uuid 前缀隔离（避免并发/重复上传同名文件互相覆盖）；
   - `scripts/cleanup_dirty_documents.py`：清理历史"双 uuid 前缀"脏文件（默认 dry-run，`--apply` 才删除）；
   - 任务队列消费者自愈：剔除死线程、按需补齐，Redis 恢复后自动重启消费。

5. **测试与评测沉淀**
   - `tests/` 新增：`rerank.py`（bge-reranker-v2-m3 独立重排服务）、`verify_dual_retrieval.py`（双路召回归入）、`test_memory.py`、`test_top5_top3.py`、`test_import.py` 等。

### 兼容性说明

- 新增环境变量：`REDIS_CACHE_TTL_SECONDS`（默认 300）、`TASK_QUEUE_MAX_ATTEMPTS` 等；
- 启动流程不变：仍为「Embedding(8002) + Rerank(8001) + 主服务(8000)」三进程。

---

## 八、目录结构（核心）

```
backend/                 # FastAPI 应用
├── agent.py             # 对话编排：并发双路召回 → 门控 → LLM 生成
├── rag_utils.py         # 检索核心：混合检索 → rerank → Auto-merging
├── rag_pipeline.py      # LangGraph：grade / rewrite(step_back/hyde/complex) / expand
├── answerability.py     # 可回答性门控四态判定（零 LLM）
├── milvus_client.py     # Milvus 客户端（dense + 内置 BM25 混合检索，RRF 融合）
├── milvus_writer.py     # 向量化入库
├── embedding.py         # embedding 客户端（HTTP 调独立服务）
├── embedding_server.py  # embedding 独立推理服务（8002）
├── document_loader.py   # 三级嵌套分块（L1/L2/L3）
├── parent_chunk_store.py# 父块（L1/L2）存储：PG + Redis 缓存
├── hooks.py / hooks_builtin.py  # Hooks 运行时与内置钩子（门控/压缩/记忆）
├── context_compact.py   # 上下文压缩（L1 预算 + 历史分级）
├── memory.py            # 跨会话记忆
├── graphkb/             # KG 统一层：GraphKB + upload_graph(Neo4j) + lightrag_kb
├── task_queue.py        # Redis Stream 任务队列（失败重试 + 线程池降级）
└── chatter/             # 颤振诊断（SAE + Stacking 融合）
kg/                      # 领域规则图谱（NetworkX，事实源 machine_fault.json + triples/）
scripts/                 # 运维脚本（重建、清理脏文件、双路召回归入验证）
tests/                   # 测试与独立服务（rerank.py）
docs/                    # 技术文档（TECH_GUIDE / RAG_STORAGE_RULES / KNOWLEDGE_GRAPH_UPGRADE 等）
```

---

> 更深入的实现细节见 `docs/TECH_GUIDE.md`（技术深度）与 `docs/RAG_STORAGE_RULES.md`（存储与图谱抽象规范）。
