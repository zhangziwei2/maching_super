# tests/

统一的测试目录。所有脚本的路径均基于 `__file__` 推导，**从任意工作目录执行都可以**。

## 目录内容

| 文件 | 作用 | 外部依赖 |
|---|---|---|
| `test_memory.py` | 记忆系统自检（建表 / 多用户隔离 / Redis 缓存 / 防注入 / 压缩兜底 / 确认写入） | 无 |
| `test_import.py` | 后端包导入检查（定位循环导入、依赖缺失） | 无 |
| `test_top5_top3.py` | RAG 检索质量评测，输出 Recall@3/5、Precision@3/5、MRR | Milvus + Rerank 服务 |
| `eval_golden.json` | 金标集（30 条问题 + 相关 chunk_id）**测试输入，入库** | — |
| `eval_report.json` | 评测报告**运行产物，已 gitignore** | — |

## 运行

```powershell
# 在项目根目录执行（也可在任意目录，路径不依赖 CWD）

# 1. 记忆系统自检 —— 无需任何外部服务，改动记忆模块后必跑
python tests/test_memory.py

# 2. 导入检查
python tests/test_import.py

# 3. RAG 评测 —— 需先启动基础设施与重排服务
docker compose up -d
python rerank.py                 # 另开一个终端
python tests/test_top5_top3.py
```

## 约定

**新增测试请统一放在本目录。** 路径一律这样取，不要依赖 CWD：

```python
from pathlib import Path
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent

# 需要 import backend.* 时，插入项目根而非脚本自身目录
sys.path.insert(0, str(_PROJECT_ROOT))

# .env 在项目根
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# 测试数据放本目录
DATA = _TESTS_DIR / "xxx.json"
```

### 为什么强调这点

`test_top5_top3.py` 原先写在项目根目录，用了三处隐含假设：

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 假设自己在根
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env") # 假设 .env 同级
GOLDEN_PATH = "eval_golden.json"                                  # 假设 CWD 是根
```

迁入 `tests/` 后这三处会同时失效：包导入失败、`.env` 读不到（表现为 `RERANK_*` 配置全空）、金标集找不到。现在已全部改为基于 `__file__` 推导。

**注意 `load_dotenv` 找不到文件时是静默失败**——不会抛异常，只会让配置读到空字符串。这类 bug 的症状是"模型调用失败"而非"路径错误"，排查成本很高。

## 评测数据说明

`eval_report.json` 的 `summary` 段是简历第 1 条的数据来源：

```
baseline:  recall@3 = 88.9%   recall@5 = 93.3%   mrr = 88.9%
reranked:  recall@3 = 93.3%   recall@5 = 97.2%   mrr = 94.4%
```

`baseline` 是**混合检索未重排**，不是单路稠密检索——引用时不要写错对比基准。
