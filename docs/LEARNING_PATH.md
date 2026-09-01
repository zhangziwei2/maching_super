# maching 深入学习路径

> 目标：从"能讲清简历五条"进阶到"能独立改造这个项目"。
> 分 5 个阶段，每阶段给出**读什么 / 要能回答什么 / 动手做什么 / 怎么验证**。

## 前置：先把环境跑起来

```powershell
# 1. 确认依赖
cd d:/Azhangziwei/AgentChef312/maching
D:\Azhangziwei\AgentChef312\.venv\Scripts\python.exe -c "import fastapi, langgraph, pymilvus, neo4j; print('ok')"

# 2. 起基础设施
docker compose up -d

# 3. 跑通现有自检
D:\Azhangziwei\AgentChef312\.venv\Scripts\python.exe test_memory.py
```

**卡点提示**：Milvus 与 Neo4j 起来较慢（首次要拉镜像），先用 `docker compose ps` 确认健康再跑应用。

---

## 阶段 1：打通主链路（1-2 天）

### 读什么

| 顺序 | 文件 | 关注点 |
|---|---|---|
| 1 | `main.py` | 应用入口、路由挂载 |
| 2 | `backend/api.py` | `/chat` 端点，看一次请求进来经过哪些步骤 |
| 3 | `backend/agent.py` | 核心：`chat_with_agent`、`_prepare_answer` |
| 4 | `backend/tools.py` | 工具定义、`CHUNK_SEPARATOR`、ContextVar 请求态 |

### 要能回答

- 一次 `/chat` 请求，从入口到返回经过了哪些函数？画出调用链。
- `set_request_identity()` 为什么必须在 `copy_context()` 之前调用？
- 流式与非流式两条路径的差异在哪？

### 动手

在 `_prepare_answer()` 里加一行日志，打印 `user_text / kg_result 长度 / rag_gate`，然后发一次请求观察输出。

### 验证

能用一句话说清："提问进来后，先路由、再检索、再门控、再压缩、最后生成"。

---

## 阶段 2：吃透 RAG 检索（2-3 天）

### 读什么

| 文件 | 关注点 |
|---|---|
| `backend/rag_utils.py` | `retrieve_documents` 是核心，读完整个函数 |
| `backend/milvus_client.py` | `hybrid_retrieve`（BM25 + dense + RRF）、`dense_retrieve` 降级 |
| `backend/document_loader.py` | 三级分块的具体切法与 `chunk_level` 赋值 |
| `backend/Intent_router.py` | 三种模式的路由判断 |
| `backend/embedding.py` | embedding 服务封装 |

### 要能回答

- `candidate_k = top_k * 3` 在哪一行？改成 5 倍会怎样？
- 为什么 `filter_expr` 限定 `chunk_level == 3`？
- RRF 融合的具体公式是什么？`k=60` 起什么作用？
- 混合检索失败后降级路径是什么？`retrieval_mode` 会变成什么值？
- Auto-merging 的触发条件是什么？上卷几层？

### 动手

1. 把 `AUTO_MERGE_ENABLED` 关掉，重跑 `test_top5_top3.py`，观察命中率变化
2. 把 Rerank 关掉，对比 `rag_gate` 的 `top1_rerank_score` 分布
3. 写一个脚本，对同一条 query 分别打印：混合检索 top15、重排后 top5、上卷后最终结果的 chunk_id 与文本前 100 字

### 验证

能解释清楚"为什么这套组合比单路稠密好"，并指出**哪一层贡献最大**（提示：用上面的对照实验数据说话）。

### 延伸阅读

`eval_golden.json`（30 题评测集）+ `eval_report.json`（结果）→ 找到生成报告的脚本（可能是 `test_top5_top3.py`），读懂评测逻辑。

---

## 阶段 3：门控与可回答性（1 天）

### 读什么

`backend/answerability.py` 全文，重点是 `evaluate()` 与 `build_rejection_message()`。

### 要能回答

- 三种判定（hard / soft / conflict）的触发条件与优先级
- 冲突检测为什么用"共享 ≥2 个主题词 + 一方否定一方肯定"？有什么反例？
- 门控为什么注册为 `critical=True`？去掉会怎样？
- 拒答文案里包含什么信息？对用户有什么用？

### 动手

1. 构造三类输入各 3 条（空检索 / 低分 / 冲突），打印 `rag_gate` 完整结果
2. 把 `SOFT_REJECT_THRESHOLD` 从 0.3 调到 0.1 和 0.5，观察拒答率变化

### 验证

能画出 `evaluate()` 的判断流程图，并说明每个分支的工业意义。

---

## 阶段 4：上下文压缩与记忆（2-3 天）

### 读什么

| 文件 | 关注点 |
|---|---|
| `backend/context_compact.py` | L1-L4 四层，`before_drop` 回调位置 |
| `backend/memory.py` | selection（`_score`）、三个写入路径、Redis 缓存 |
| `backend/hooks.py` | 钩子机制、`critical` 语义 |
| `backend/hooks_builtin.py` | 注册了什么、`context_pipeline` 为什么是管道 |
| `backend/models.py` | `Memory`、`ContextBlob`、`ChatTranscript` |

### 要能回答

- L1-L4 各解决什么问题？参数分别是多少？
- **为什么 `context_budget` 和 `memory_injection` 必须聚合成 pipeline？**（这是全项目最容易讲错的点）
- 为什么记忆注入必须在压缩之后？
- `before_drop` 在代码里的确切位置？如果移到摘要之后会怎样？
- `_score` 的打分顺序：`score <= 0` 的返回为什么在 `_MACHINE_BONUS` 之前？
- 机器级记忆为什么走 Redis 而个人记忆不走？
- 抽取 prompt 的防注入声明在哪一句？去掉会有什么风险？

### 动手

1. **先跑通 `test_memory.py`**（这是当前唯一未验证的模块）
2. 造一个超长会话（连续提问 20 轮），观察 L2/L3/L4 分别在什么时候触发，打印每层的输入输出长度
3. 造一条含设备参数的对话，触发压缩，确认记忆是否真的接住了（查 `memories` 表）
4. 调用 `POST /memories` 写一条机器级档案，再用另一个账号查询，确认可见；然后观察 Redis 里 `memory:machine` 的失效时机

### 验证

能完整讲清"压缩和记忆如何配合"，包括那个关键追问：**"兜底抽取不还是推断吗？"**

---

## 阶段 5：架构与扩展（持续）

### 读什么

- `backend/task_queue.py` —— Redis Stream 任务队列
- `backend/upload_jobs.py` / `milvus_writer.py` —— 文档入库与增量写入
- `backend/context_compact.py` 的 `_archive` 目录对照 —— 看旧版本实现，理解演进
- `docs/TECH_PLAN_HARNESS.md` —— 原始设计方案
- `docs/TECH_GUIDE.md` —— 本文档配套的简历技术点展开

### 要能回答

- 文档上传后经过哪些步骤进入 Milvus？增量写入如何检测变更？
- 任务队列为什么用 Redis Stream 而非 Celery？
- 对比 `_archive/` 里的旧实现，现在的版本改进了什么？
- 如果让你加一个"查询改写"能力，你会挂在哪里？怎么写？

### 动手（任选）

| 任务 | 难度 | 涉及 |
|---|---|---|
| 实现记忆 consolidation（≥10 条合并） | 中 | `memory.py` |
| 实现 `pending` 审核态 + admin 确认 | 中 | `models.py`、`memory.py`、`api.py` |
| 实现 T3 能力画像（追问次数统计，零 LLM） | 低 | `memory.py` |
| 让记忆参与门控判定 | 中 | `answerability.py`、`hooks_builtin.py` |
| 补 S1 门控收编对照用例 | 低 | 新建测试 |

### 验证标准

独立完成上表任一任务，且 `test_memory.py` 仍全绿。

---

## 速查：关键常量位置

| 常量 | 文件:行 | 值 |
|---|---|---|
| `LARGE_CHUNK_LIMIT` | `context_compact.py:44` | 8000 |
| `LONG_MESSAGE_LIMIT` | `context_compact.py:45` | 4000 |
| `PREVIEW_CHARS` | `context_compact.py:46` | 1000 |
| `KEEP_RECENT_TURNS` | `context_compact.py:47` | 3 |
| `CONTEXT_CHAR_LIMIT` | `context_compact.py` | 50000 |
| `SOFT_REJECT_THRESHOLD` | `answerability.py` | 0.3 |
| `LEAF_RETRIEVE_LEVEL` | `rag_utils.py` | 3 |
| `AUTO_MERGE_THRESHOLD` | `rag_utils.py` | 2 |
| `candidate_k` | `rag_utils.py:249` | top_k × 3 |
| RRF k | `milvus_client.py` | 60 |
| `MACHINE_CACHE_KEY` | `memory.py` | `memory:machine` |
| `EXTRACT_TYPES` | `memory.py` | `("user", "project")` |
| `CHUNK_SEPARATOR` | `tools.py:29` | `"\n\n---\n\n"` |

> 行号会随代码变动，以常量名为准搜索。

## 学习建议

1. **边读边画**：每个模块读完画一张数据流图，比记代码有效
2. **改参数观察**：这项目的开关都做成环境变量了，改参数看行为变化是最快的理解方式
3. **先跑测试再改代码**：`test_memory.py` 是你唯一的回归网，改之前确认它全绿
4. **对照 harness**：`D:\learn_clcude\learn-claude-code\s08_context_compact` 和 `s09_memory` 是原始参考，对比差异能加深理解"为什么这么改"
