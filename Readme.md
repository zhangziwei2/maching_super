# Maching — 机床故障诊断知识图谱 + RAG 智能问答系统

面向**机床智能诊断**工业场景的检索增强问答系统。以「知识图谱精准检索 + 文档 RAG 语义检索」双通道为核心，配合查询改写、结果重排、自动父块合并与**可回答性门控**，在保证答案可溯源的前提下降低幻觉；工程侧通过模型服务解耦、任务队列与全异步化，支撑多用户并发与长时间入库任务。系统同时兼容医疗、教育等行业知识库场景。

---

## 一、核心功能

### 1. 双通道混合检索（核心定位）
- **知识图谱通道（KG）**：基于内置图引擎（NetworkX + AC 自动机实体锚定），支持 1~2 跳子图 BFS 扩展，输出结构化三元组与故障属性（类型/原因/处理建议），答案确定性强、可解释。
- **文档 RAG 通道（Milvus）**：
  - **混合检索**：dense 向量（bge-m3）+ Milvus 原生 BM25 稀疏向量，RRF（k=60）融合，兼顾语义匹配与关键词精确命中；
  - **Cross-Encoder 重排**：`bge-reranker-v2-m3` 对候选片段精排，提升 Top-K 质量；
  - **Auto-merging 检索**：三级分块（L1/L2/L3），先检索叶子块再按命中数自动上卷合并至父块，兼顾检索精度与上下文完整性。

### 2. 智能意图路由（零额外 LLM 开销）
纯规则预路由：命中 KG 强信号词仅走图谱、命中 RAG 信号词仅走文档库、未命中或双命中则双通道并发融合。**不额外消耗 LLM 调用**，显著降低延迟与成本。

### 3. 可回答性门控（抗幻觉，关键特色）
在生成答案前对检索结果做组合判定，避免"无依据硬答"：
- **硬拒答**：检索结果为空 → 明确告知知识库无相关内容，不臆造答案；
- **软拒答**：Top-1 重排分低于阈值（默认 0.3）→ 输出「可能原因 + 置信度 + 依据缺失项」，引导用户补充现场信息；
- **冲突检测**：多来源结论矛盾 → 提示矛盾点并请求更权威依据，不强行二选一。

该门控为纯规则实现，**零额外 LLM 调用**，并将判定结果透出至 `rag_trace` 便于审计。

### 4. 检索过程全链路可观测
RAG 流水线基于 LangGraph 编排，完整记录 `retrieve → grade → rewrite → 检索扩展 → merge → rerank` 各阶段指标与耗时，前端可展示逐步执行过程与引用来源，便于效果调优与问题定位。

### 5. 文档管理与异步入库
- 支持 PDF / Word / TXT / Markdown 上传，自动完成解析、三级分块、父块入库 PG、叶子向量化入库 Milvus，并可选自动抽取实体关系图谱；
- 上传/删除任务由 **Redis Stream 任务队列**调度，支持失败自动重试（最多 3 次，指数退避），Redis 不可用时自动降级线程池；
- 任务状态持久化于数据库（7 天 TTL + 看门狗兜底），服务重启不丢失，前端轮询展示分步进度。

### 6. 其他能力
- **用户与会话体系**：JWT 认证、角色区分（管理员可管理文档/图谱），多会话历史持久化。
- **颤振诊断**：内置基于振动信号多层融合（数据层→特征层→决策层）的颤振自动诊断模块，支持 CSV/XLSX 数据分析与实时监测报警。

---

## 二、环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.12 | 项目声明 `requires-python = ">=3.12"` |
| Docker / Docker Compose | 最新稳定版 | 用于启动依赖中间件 |
| PostgreSQL | 15（容器提供） | 父块存储、会话与任务状态 |
| Redis | 7（容器提供） | 缓存与任务队列（Stream） |
| Milvus | 2.5.x（容器提供） | 混合检索向量库，需支持原生 BM25 Function |
| Neo4j | 5（容器提供，可选） | LightRAG 图谱抽取 |
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

在项目根目录创建 `.env`（以下为关键项，其余均有默认值）：

```ini
# ===== LLM =====
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# ===== 数据库 / 缓存 =====
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app
REDIS_URL=redis://localhost:6379/0

# ===== 向量库 =====
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=embeddings_collection

# ===== 认证 =====
JWT_SECRET_KEY=change-this-secret
ADMIN_INVITE_CODE=your_admin_invite_code

# ===== 模型服务（第二轮迭代新增）=====
EMBEDDING_SERVICE_URL=http://127.0.0.1:8002
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
EMBEDDING_BINDING_PORT=8002
RERANK_BINDING_HOST=http://127.0.0.1:8001
RERANK_DEVICE=cuda
EMBEDDING_API_KEY=          # 可选，设置后服务将校验 Bearer Token

# ===== 检索调优（可选）=====
AUTO_MERGE_ENABLED=true
ANSWERABILITY_RERANK_THRESHOLD=0.3   # 可回答性软拒答阈值
TASK_QUEUE_MAX_ATTEMPTS=3            # 任务队列最大重试次数

# ===== 上传限制（第三轮迭代新增）=====
MAX_UPLOAD_MB=200                    # 单个上传文件大小上限（MB）
```

### 4. 启动模型推理服务（两个独立进程）

第二轮迭代已将模型推理从 API 进程解耦，需**单独启动**以下两个服务：

```bash
# 终端 1：Embedding 服务（端口 8002，bge-m3）
python -m backend.embedding_server

# 终端 2：Rerank 服务（端口 8001，bge-reranker-v2-m3）
python rerank.py
```

服务就绪后可通过 `http://127.0.0.1:8002/health` 验证 embedding 服务状态。

> 模型仅在独立进程中加载一次，API 进程纯 HTTP 调用；首次启动会自动下载模型（bge-m3 约 2GB）。

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

> 旧版本使用自实现 BM25（本地 JSON 状态），与新 schema 不兼容，必须重建后才能正常检索。

### 7. 训练颤振诊断模型【可选】

```bash
python -m backend.chatter.train_all
```

---

## 四、第二轮迭代更新

本轮迭代围绕**「抗幻觉、降延迟、提可用性」**三条主线，共完成 6 项新增与升级。

### 新增功能

1. **可回答性门控（`backend/answerability.py`）**
   新增空检索硬拒答、低置信度软拒答（输出「可能原因 + 置信度 + 依据缺失项」）、多文档结论冲突检测三重机制，纯规则实现、零额外 LLM 调用；判定结果写入 `rag_trace` 并对前端透出。

2. **Embedding 独立推理服务（`backend/embedding_server.py`）**
   新增端口 8002 的 HTTP 推理服务（`/v1/embeddings`，OpenAI 兼容格式），支持可选 API Key 鉴权与 `/health` 健康检查。

3. **Redis Stream 任务队列（`backend/task_queue.py`）**
   新增基于 `XADD` / `XREADGROUP` / `XACK` 的任务调度，支持失败自动重试 3 次（指数退避 1s/2s/4s），并在 Redis 不可用时自动降级线程池，保证入库任务不丢失、不阻塞。

4. **Milvus 全量重灌脚本（`backend/rebuild_milvus.py`）**
   新增集合重建脚本，从 PostgreSQL 全量恢复分块数据，支持 `--yes` 非交互执行，用于 schema 升级后的数据迁移。

### 功能升级

5. **稀疏检索迁移至 Milvus 原生 BM25**
   移除自实现的 BM25（本地 JSON 词表与词频状态），改为在 Milvus schema 中注册原生 `FunctionType.BM25`（text 字段启用中文 analyzer），稀疏索引切换为 `metric_type="BM25"`。
   **收益**：消除多进程间词表不一致与状态文件损坏风险，IDF 统计由服务端自动维护，删除数据后仅需 `flush` 即可刷新，彻底解决旧方案"单文件状态"的可靠性瓶颈。

6. **Embedding 服务解耦与主接口全异步化**
   - `backend/embedding.py` 精简为纯 HTTP 客户端，API 进程不再加载模型、无本地兜底，模型只驻留独立进程，支持独立扩容与重启；
   - 主聊天接口 `chat_endpoint` 改用 `asyncio.to_thread`，流式链路中的同步 LLM 调用移入线程池，避免阻塞 FastAPI 事件循环；
   - 意图路由重写为纯规则实现（去除原有模型依赖），消除冗余开销，并按「命中单通道仅走该通道、未命中双通道并发」策略调度，减少无效检索。

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

8. **移除联网搜索兜底**
   删除 `_eval_sufficiency`、`_search_web`、`web_search` 工具及相关 prompt。
   - **省一次 LLM 调用**：原每次对话都额外调用 LLM 判断「是否需要联网」；
   - **消除逻辑矛盾**：该判断规则中「参考资料为空 → SUFFICIENT（让模型自行发挥）」与第二轮引入的可回答性门控（依据不足应拒答）直接冲突。机床场景为封闭领域，联网价值有限，移除后行为更一致。

9. **统一文件上传逻辑**
   原 4 处重复的 1MB 分块写盘循环（`_save_chatter_upload`、`_save_upload_file`、`/upload/image` 内联、`/kg/triples/import` 内联）合并为单个 `_save_upload_stream()`，消除重复代码，并统一获得路径净化与大小限制能力。

### 兼容性说明

- **行为变更**：`/upload/image` 返回的 `filename` 字段现包含 uuid 前缀（如 `a1b2c3d4_photo.png`），如需展示原名请自行截取或告知调整；
- **依赖清理**：联网搜索移除后 `tavily-python` 不再需要，可从 `requirements.txt` 删除；
- **新增环境变量**：`MAX_UPLOAD_MB`（上传大小上限，默认 200）；
- **数据/启动无变更**：本轮为纯代码层修复，**无需**重建向量集合或改动启动流程。
