"""全量双路召回归入验证（不需要真实服务也可跑前两组）

三组检查：
  1. 路由判定           —— 纯规则，零依赖
  2. 上下文依据标注     —— 纯函数，零依赖
  3. 门控判定           —— 纯函数，零依赖

第 4 组（端到端实跑）需要 Milvus/PG/Redis 与 LLM，默认跳过，
加 --live 参数才执行。

运行：
  python scripts/verify_dual_retrieval.py
  python scripts/verify_dual_retrieval.py --live      # 含端到端，需外部服务
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


# ============ 1. 路由判定 ============
print("== 1. 路由判定（纯规则） ==")
try:
    from backend.Intent_router import IntentRouter
except ImportError as e:
    check("IntentRouter 可导入", False, repr(e))
    sys.exit(1)

router = IntentRouter()

# 这是本次要修复的原始问题：含「主轴」但问的是手册章节内容
CASES = [
    # (问题, 期望是否命中 RAG 信号)
    ("主轴安全说明 2.1 注意事项有哪些？", True),
    ("电主轴严禁哪些操作？", True),
    ("主轴的技术参数是多少？", True),
    ("主轴异响怎么处理？", False),      # 纯故障类，应无 RAG 信号
]

for q, expect_rag in CASES:
    r = router.route(q)
    rag_matched = [kw for kw in router.RAG_KEYWORDS if kw in q]
    hit = len(rag_matched) > 0
    check(f"路由-RAG信号「{q[:18]}」", hit == expect_rag,
          f"route={r['route']} 命中={rag_matched[:3]}")

# 关键词补全验证：这些词此前缺失，是漏召回的直接诱因
for kw in ["安全", "注意事项", "严禁", "技术参数"]:
    check(f"关键词已收录「{kw}」", kw in router.RAG_KEYWORDS)

# ============ 2. 上下文依据标注 ============
print("\n== 2. 上下文依据标注 ==")
try:
    from backend.agent import _build_context
    check("_build_context 可导入", True)
except ImportError as e:
    check("_build_context 可导入", False, repr(e))
    sys.exit(1)

# 场景 A：两路都有内容 —— 都应出现并标注来源
ctx_both = _build_context("主轴-[故障]->异响", "非我司指定的技术人员禁止拆卸本电主轴。")
check("两路-含图谱来源标注", "知识图谱检索结果" in ctx_both)
check("两路-含文档来源标注", "文档知识库检索结果" in ctx_both)
check("两路-文档标注优先采用", "优先采用" in ctx_both)
check("两路-不触发依据提示", "依据提示" not in ctx_both)

# 场景 B：仅图谱命中 —— 必须出现依据提示，这是防编造的关键
ctx_kg_only = _build_context("主轴-[故障]->异响", "")
check("仅图谱-含依据提示", "依据提示" in ctx_kg_only)
check("仅图谱-提示勿凭通用知识补充", "不要凭通用知识补充" in ctx_kg_only)
check("仅图谱-不含文档段", "文档知识库检索结果" not in ctx_kg_only)

# 场景 C：仅文档命中 —— 不应出现依据提示
ctx_rag_only = _build_context("", "非我司指定的技术人员禁止拆卸。")
check("仅文档-含文档来源标注", "文档知识库检索结果" in ctx_rag_only)
check("仅文档-不触发依据提示", "依据提示" not in ctx_rag_only)

# 场景 D：两路皆空
check("两空-返回空串", _build_context("", "") == "")

# ============ 3. 门控判定 ============
print("\n== 3. 门控判定（全量双路版本）==")
try:
    from backend.hooks_builtin import answerability_gate
    check("answerability_gate 可导入", True)
except ImportError as e:
    check("answerability_gate 可导入", False, repr(e))
    sys.exit(1)

# 关键回归：旧逻辑「图谱非空即放行」会让门控永不生效。
# 新逻辑下，图谱非空 + 文档空 仍放行（依据标注接管防编造），
# 但两源皆空必须硬拒答。
check("门控-两源皆空硬拒答",
      answerability_gate("主轴安全说明 2.1 注意事项", "", "", {}) is not None)

# 文档有内容 + 门控 pass → 放行
check("门控-文档pass放行",
      answerability_gate("主轴异响", "图谱", "文档", {"status": "pass"}) is None)

# 文档有内容 + soft_reject → 拒答（即使图谱非空，也必须拒）
gate_soft = {"status": "soft_reject", "confidence": 0.1, "missing": ["故障工况"]}
check("门控-低分拒答(图谱非空也拒)",
      answerability_gate("主轴异响", "图谱有内容", "文档", gate_soft) is not None)

# 文档有内容 + conflict → 提示矛盾
check("门控-冲突提示",
      answerability_gate("主轴异响", "图谱", "文档",
                         {"status": "conflict", "conflict_reason": "结论相反"}) is not None)

# 仅图谱命中 → 放行（依据标注已在 context 中提示模型）
check("门控-仅图谱放行",
      answerability_gate("主轴注意事项", "图谱有内容", "", {}) is None)

# ============ 4. 端到端（可选） ============
if "--live" in sys.argv:
    print("\n== 4. 端到端实跑 ==")
    try:
        from backend.agent import _retrieve_all
        kg, rag, gate, trace = _retrieve_all("主轴安全说明 2.1 注意事项有哪些？")
        print(f"  KG 长度={len(kg)}  RAG 长度={len(rag)}")
        print(f"  route={trace.get('route') if trace else None}")
        print(f"  gate={gate}")
        # 核心断言：文档通道必须被真正访问（旧逻辑下此处必为空）
        check("实跑-文档通道已检索", len(rag) > 0,
              "RAG 为空说明文档库未命中，请确认文档已入库")
        check("实跑-含文档原文特征", ("禁止" in rag or "注意事项" in rag or "电主轴" in rag),
              rag[:80])
    except Exception as e:
        check("实跑-执行", False, repr(e))
else:
    print("\n== 4. 端到端实跑 == 已跳过（加 --live 执行，需 Milvus/PG/Redis）")

print()
print(("FAILED: " + ", ".join(failures)) if failures else "ALL PASSED")
