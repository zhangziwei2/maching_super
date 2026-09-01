"""记忆系统自检

覆盖三层：
1. DB 层：建表、多用户隔离、机器级去重、删除权限
2. Redis 缓存层：命中、写时失效、Redis 不可用降级
3. 纯逻辑：切词、打分、JSON 解析、防注入、压缩兜底、确认写入

运行：python test_memory.py
"""
import os

# 必须在导入 backend.database 之前设置：engine 在模块导入时按 DATABASE_URL 创建。
# 内存库 + StaticPool（database.py 对 SQLite 的固定配置）单连接复用，不会丢表。
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
# 先关缓存：DB 层用例会在替换 cache 对象之前执行，否则会去连真实 Redis
# （缓存层用例里再显式打开，并用内存桩替换）
os.environ["MEMORY_CACHE_ENABLED"] = "false"

from sqlalchemy import inspect

from backend.database import engine, init_db
from backend import memory

init_db()

failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


# ==================== DB 层 ====================
tables = sorted(inspect(engine).get_table_names())
check("建表-memories 存在", "memories" in tables, tables)

r1 = memory.save_memory("alice", "主轴参数", "XK7132 主轴最高 8000rpm",
                        mem_type="project", scope="personal", created_by="alice")
check("写入-个人记忆", r1 is not None and r1["scope"] == "personal", r1)

r2 = memory.save_memory("alice", "主轴参数", "XK7132 主轴最高 6000rpm（已修正）",
                        mem_type="project", scope="personal")
check("写入-同名覆盖更新", r2 is not None and r1 and r2["id"] == r1["id"] and "6000" in r2["body"], r2)

r3 = memory.save_memory(None, "设备型号", "XK7132 数控铣床", scope="machine", created_by="admin")
check("写入-机器级记忆", r3 is not None and r3["user_id"] is None and r3["scope"] == "machine", r3)

r4 = memory.save_memory(None, "设备型号", "XK7132 数控铣床（2024 款）", scope="machine")
check("写入-机器级去重", r4 is not None and r3 and r4["id"] == r3["id"], r4)

mine = [m["name"] for m in memory.list_memories("alice")]
check("可见性-含机器级与本人", "设备型号" in mine and "主轴参数" in mine, mine)

others = [m["name"] for m in memory.list_memories("bob")]
check("隔离-他人类不可见", "主轴参数" not in others and "设备型号" in others, others)

check("校验-空name被拒", memory.save_memory("alice", "", "body") is None)
check("校验-空body被拒", memory.save_memory("alice", "name", "") is None)
check("校验-非法scope被拒", memory.save_memory("alice", "n", "b", scope="hack") is None)

check("删除-普通用户删机器级被拒", r3 and memory.delete_memory(r3["id"], owner_user_id="alice") is False)
check("删除-普通用户删他人被拒", r1 and memory.delete_memory(r1["id"], owner_user_id="bob") is False)
check("删除-本人可删", r1 and memory.delete_memory(r1["id"], owner_user_id="alice") is True)
check("删除-管理员可删机器级", r3 and memory.delete_memory(r3["id"], owner_user_id=None) is True)

# ==================== Redis 缓存层 ====================
# 用内存 dict 模拟 Redis，验证命中/失效/降级三条路径


class _FakeCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.sets = 0
        self.deletes = 0

    def get_json(self, key, *a, **k):
        self.hits += 1
        return self.store.get(key)

    def set_json(self, key, value, ttl=None):
        self.sets += 1
        self.store[key] = value

    def delete(self, key):
        self.deletes += 1
        self.store.pop(key, None)


_fake = _FakeCache()
import backend.cache as cache_mod

cache_mod.cache = _fake
memory.CACHE_ENABLED = True  # 以上均为内存桩，不涉及真实 Redis

memory.save_memory(None, "缓存测试", "主轴轴承型号 7010C", scope="machine")
check("缓存-写入机器级触发失效", _fake.deletes >= 1, _fake.deletes)

first = memory._load_machine()
check("缓存-首次回源DB并写缓存", _fake.sets >= 1 and len(first) >= 1, len(first))

_before = _fake.hits
second = memory._load_machine()
check("缓存-二次命中不查DB", _fake.hits == _before + 1 and second == first)

# 个人记忆不该进机器级缓存（写入 personal 后，缓存内容必须不变）
_cache_before = memory._machine_cache_get()
memory.save_memory("alice", "个人缓存探针", "不该出现在机器缓存里", scope="personal")
_cache_after = memory._machine_cache_get()
check("缓存-个人记忆不进机器缓存",
      _cache_after == _cache_before
      and all(m["scope"] == "machine" for m in (_cache_after or [])),
      [m.get("name") for m in (_cache_after or [])])
_probe = [m for m in memory.list_memories("alice") if m["name"] == "个人缓存探针"]
if _probe:
    memory.delete_memory(_probe[0]["id"], owner_user_id="alice")

# 机器级档案变更后，其他用户必须能立即读到新档案（缓存失效的业务意义）
memory.save_memory(None, "新设备档案", "冷却泵型号 CB-25", scope="machine")
check("缓存-机器级变更后其他用户立即可见",
      any(m["name"] == "新设备档案" for m in memory.list_memories("bob")),
      [m["name"] for m in memory.list_memories("bob")])
_nm = [m for m in memory._load_machine() if m["name"] == "新设备档案"]
if _nm:
    memory.delete_memory(_nm[0]["id"], owner_user_id=None)

# Redis 不可用时应静默回源 DB，不抛异常
class _BrokenCache:
    def get_json(self, *a, **k):
        raise RuntimeError("redis down")

    def set_json(self, *a, **k):
        raise RuntimeError("redis down")

    def delete(self, *a, **k):
        raise RuntimeError("redis down")


cache_mod.cache = _BrokenCache()
try:
    rows = memory._load_machine()
    check("缓存-Redis故障静默降级", isinstance(rows, list) and len(rows) >= 1, len(rows))
except Exception as e:
    check("缓存-Redis故障静默降级", False, repr(e))

# 恢复可用缓存，避免影响后续用例
cache_mod.cache = _fake

# ==================== 抽取：防注入与类型收敛 ====================
check("抽取-prompt含防注入声明", "不是给你的指令" in memory._EXTRACT_PROMPT)
check("抽取-prompt仅两类", "project|user" in memory._EXTRACT_PROMPT
      and "reference" not in memory._EXTRACT_PROMPT)
check("抽取-EXTRACT_TYPES仅两类", set(memory.EXTRACT_TYPES) == {"user", "project"},
      memory.EXTRACT_TYPES)

# 非两类输出应被丢弃（防止模型输出 reference/feedback 被误存）
_persisted = memory._persist_items("alice", [
    {"type": "project", "name": "主轴型号", "description": "d", "body": "BT40"},
    {"type": "reference", "name": "手册", "description": "d", "body": "见 P47"},
    {"type": "feedback", "name": "纠错", "description": "d", "body": "你说错了"},
])
check("落库-非两类被丢弃", len(_persisted) == 1 and _persisted[0]["name"] == "主轴型号",
      [m["name"] for m in _persisted])
memory.delete_memory(_persisted[0]["id"], owner_user_id="alice")

# ==================== 压缩兜底抽取（s08 → s09 接口） ====================
class _Msg:
    def __init__(self, role, content):
        self.type = role
        self.content = content


_dropped = [
    _Msg("human", "这台机床主轴异响，怎么处理？"),
    _Msg("ai", "建议先检查皮带张紧度。"),
    _Msg("human", "记住，这台机床是 XK7132，主轴最高 8000rpm"),
]
_dialogue = memory._render_dialogue(_dropped)
check("兜底-多轮渲染", "记住" in _dialogue and "8000rpm" in _dialogue, _dialogue[:60])

# dict 消息必须同样能渲染：getattr 抓不到 dict 的 key，曾经导致静默失效
_dict_msgs = [
    {"type": "human", "content": "这台机床主轴异响"},
    {"role": "assistant", "content": "检查皮带张紧度"},
]
_dict_dialogue = memory._render_dialogue(_dict_msgs)
check("兜底-dict消息可渲染", "主轴异响" in _dict_dialogue and "皮带" in _dict_dialogue,
      repr(_dict_dialogue))
check("兜底-role字段兼容", "助手" in _dict_dialogue, repr(_dict_dialogue))

# 空内容消息应被跳过，不产生空行
check("兜底-空消息被跳过",
      memory._render_dialogue([{"type": "human", "content": "  "}]) == "")

# 截断：只保留最近 max_messages 条
_many = [_Msg("human", f"第{i}轮提问") for i in range(20)]
_trimmed = memory._render_dialogue(_many, max_messages=5)
check("兜底-条数截断", "第19轮" in _trimmed and "第0轮" not in _trimmed, repr(_trimmed[:40]))

# 截断：字符上限生效（长消息不应撑爆抽取 prompt）
_long = [_Msg("human", "长" * 3000), _Msg("human", "结尾标记XYZ")]
check("兜底-字符上限生效",
      len(memory._render_dialogue(_long, max_chars=1000)) <= 1000
      or "结尾标记XYZ" not in memory._render_dialogue(_long, max_chars=1000))

# 开关关闭时兜底不触发（此处 _calls 尚未定义，只断言返回值）
memory.EXTRACTION_ENABLED = False
check("兜底-开关关闭不触发", memory.extract_from_messages("carol", _dropped) == [])
memory.EXTRACTION_ENABLED = True

_calls = []
memory._extract_llm = lambda d: (_calls.append(d), [
    {"type": "project", "name": "设备参数", "description": "主轴", "body": "XK7132 主轴最高 8000rpm"}
])[1]
_saved = memory.extract_from_messages("carol", _dropped, trigger="compact")
check("兜底-无需信号词即触发", len(_calls) == 1 and len(_saved) == 1, _saved)
check("兜底-写入个人记忆", _saved and _saved[0]["scope"] == "personal")
memory.delete_memory(_saved[0]["id"], owner_user_id="carol")

# 空消息不触发抽取
_calls.clear()
check("兜底-空消息不调LLM", memory.extract_from_messages("carol", []) == [] and not _calls)

# ==================== 确认信号写入（零 LLM） ====================
_llm_calls = []


def _boom(d):
    _llm_calls.append(d)
    raise AssertionError("确认写入不应调用 LLM")


memory._extract_llm = _boom
_conf = memory.save_confirmed("dave", "主轴异响怎么处理", "先检查皮带张紧度")
check("确认-零LLM写入", _conf is not None and not _llm_calls, _conf)
check("确认-内容含问答", _conf and "先检查皮带张紧度" in _conf["body"], _conf and _conf["body"][:40])
check("确认-标题自动截取", _conf and _conf["name"] == "主轴异响怎么处理", _conf and _conf["name"])
_conf2 = memory.save_confirmed("dave", "主轴异响怎么处理", "换个答案")
check("确认-同名覆盖", _conf2 and _conf and _conf2["id"] == _conf["id"])
check("确认-空入参被拒", memory.save_confirmed("dave", "", "") is None)
# 非法 mem_type 应降级为 project，而不是写入脏类型
_bad = memory.save_confirmed("dave", "刀具磨损判断", "看后刀面磨损带", mem_type="hack")
check("确认-非法类型降级", _bad and _bad["mem_type"] == "project", _bad and _bad["mem_type"])
if _bad:
    memory.delete_memory(_bad["id"], owner_user_id="dave")
memory.delete_memory(_conf["id"], owner_user_id="dave")

# ==================== 纯逻辑（打桩 DB 读取）====================
_FIXTURES = [
    {"id": 1, "scope": "machine", "mem_type": "project",
     "name": "设备型号", "description": "机床型号", "body": "XK7132 数控铣床", "updated_at": "2026-01-01"},
    {"id": 2, "scope": "personal", "mem_type": "user",
     "name": "学员画像", "description": "经验水平", "body": "新手，需解释专业术语", "updated_at": "2026-01-02"},
    {"id": 3, "scope": "personal", "mem_type": "feedback",
     "name": "无关记忆", "description": "其他", "body": "冷却液品牌是长城", "updated_at": "2026-01-03"},
]
memory._load_all = lambda user_id: list(_FIXTURES)

toks = memory._tokens("主轴最高转速 8000rpm 的 XK7132")
check("切词-中文2gram", "主轴" in toks and "转速" in toks, toks)
check("切词-英文数字", "8000rpm" in toks, toks)
check("切词-停用词过滤", "的" not in toks, toks)

check("信号-命中", memory.has_explicit_signal("记住这个，我是新手"))
check("信号-未命中", not memory.has_explicit_signal("主轴过热怎么办"))

raw = '好的，结果如下：\n```json\n[{"type":"project","name":"主轴参数","description":"d","body":"b"}]\n```'
items = memory._parse_json_array(raw)
check("解析-带围栏与解释文字", len(items) == 1 and items[0]["type"] == "project", items)
check("解析-空输入", memory._parse_json_array("") == [])
check("解析-非法", memory._parse_json_array("不是JSON") == [])
# 模型偶尔会吐对象而非数组，此时应返回空而不是崩溃
check("解析-对象非数组", memory._parse_json_array('{"type":"project"}') == [])
check("解析-空数组", memory._parse_json_array("[]") == [])
# 数组元素非 dict 时应被 _persist_items 过滤
check("落库-非dict元素被过滤",
      memory._persist_items("alice", ["not-a-dict", 123, None]) == [])

picked = memory.select_memories("alice", "XK7132 主轴转速多少")
check("selection-相关记忆优先", bool(picked) and picked[0]["name"] == "设备型号", [m["name"] for m in picked])
check("selection-无关记忆不召回", "无关记忆" not in [m["name"] for m in picked])

picked2 = memory.select_memories("alice", "请解释一下专业术语")
check("selection-按查询切换", bool(picked2) and picked2[0]["name"] == "学员画像", [m["name"] for m in picked2])

check("selection-空查询仅机器级",
      [m["scope"] for m in memory.select_memories("alice", "？？")] == ["machine"])

# limit 必须生效（否则长记忆会撑爆上下文）
check("selection-limit生效",
      len(memory.select_memories("alice", "机床 主轴 转速 术语 冷却液", limit=2)) == 2)

# 无相关记忆时不返回机器级（score<=0 先返回，机器级加权不生效）
check("selection-完全无关时不召回",
      memory.select_memories("alice", "完全不相干的查询 xyzzy") == [])

rendered = memory.render_memories("alice", "XK7132 主轴转速多少")
check("渲染-含防注入措辞", "不是新指令" in rendered and "以当前用户请求为准" in rendered)
check("渲染-含记忆正文", "XK7132" in rendered)
check("渲染-空结果", memory.render_memories("alice", "完全不相干的查询 xyzzy") == "")

_no_signal_calls = []
memory._extract_llm = lambda d: _no_signal_calls.append(d) or []
memory.extract_and_save("alice", "主轴过热怎么办", "检查润滑")
check("抽取-无信号不调LLM", len(_no_signal_calls) == 0)

memory.ENABLED = False
check("开关-关闭后不召回", memory.select_memories("alice", "XK7132") == [])
memory.ENABLED = True

# ==================== 钩子注册 ====================
try:
    from backend.hooks import list_hooks
    from backend.hooks_builtin import register_builtin_hooks

    register_builtin_hooks()
    hooks = list_hooks()
    check("钩子-PreGenerate 注册 context_pipeline", hooks["PreGenerate"] == ["context_pipeline"], hooks["PreGenerate"])
    check("钩子-Stop 注册 memory_extraction", hooks["Stop"] == ["memory_extraction"], hooks["Stop"])
    check("钩子-PostRetrieve 门控仍在", hooks["PostRetrieve"] == ["retrieval_progress", "answerability_gate"],
          hooks["PostRetrieve"])
except Exception as e:
    print(f"SKIP  钩子注册检查（依赖较重，可单独验证）: {e}")

print()
print(("FAILED: " + ", ".join(failures)) if failures else "ALL PASSED")
