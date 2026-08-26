# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱 - Schema 定义
统一实体/关系类型常量、命名空间常量、属性规范。

设计说明（借鉴 Yuxi 的 kb_id 标签隔离思想）：
- 规则图谱（来自 machine_fault.json）与人工导入图谱（来自 triples/*.jsonl）
  共存于同一个 NetworkX 图对象，通过 namespace 属性区分。
- 命名空间隔离保证：人工导入的错误数据可整体回滚（删除对应 JSONL 文件后重建）。
"""

# ========== 命名空间 ==========
NAMESPACE_RULE = "Rule"       # 规则图谱（machine_fault.json）
NAMESPACE_UPLOAD = "Upload"   # 人工/专家导入图谱（triples/*.jsonl）

ALL_NAMESPACES = [NAMESPACE_RULE, NAMESPACE_UPLOAD]

# ========== 规则图谱节点类型（英文标签，对齐 question_parser 历史模板） ==========
RULE_NODE_TYPES = [
    "FaultType",    # 故障类型：name, desc, easy_get, cure_lasttime, cured_prob
    "Cause",        # 故障原因：text
    "Symptom",      # 故障现象：text
    "Solution",     # 解决方法：text
    "Component",    # 机床部件：text
    "Prevent",      # 预防措施：text
    "Check",        # 检测方法：text
    "Material",     # 加工材料：text
    "Category",     # 故障类别：text
    "CureWay",      # 修复方式：text
    "Parameter",    # 加工参数：param_name, value, adjustment
]

# ========== 规则图谱关系类型（对齐 question_parser 历史模板） ==========
RULE_RELATIONS = [
    "HAS_CAUSE",            # FaultType -> Cause
    "HAS_SYMPTOM",          # FaultType -> Symptom
    "HAS_SOLUTION",         # FaultType -> Solution
    "INVOLVES_COMPONENT",   # FaultType -> Component
    "HAS_PREVENT",          # FaultType -> Prevent
    "HAS_PARAMETER",        # FaultType -> Parameter
    "NEEDS_CHECK",          # FaultType -> Check
    "APPLIES_TO_MATERIAL",  # FaultType -> Material
    "BELONGS_TO_CATEGORY",  # FaultType -> Category
    "HAS_CURE_WAY",         # FaultType -> CureWay
]

# 规则图谱中 FaultType 的属性字段
FAULT_TYPE_PROPS = ["name", "desc", "easy_get", "cure_lasttime", "cured_prob"]

# ========== 人工导入图谱 ==========
# Upload 实体通过 type 属性承载自定义实体类型；关系类型由边属性 rel_type 承载。
# 允许的关系类型（可扩展）：RELATED_TO 表示一般关联
UPLOAD_GENERIC_RELATION = "RELATED_TO"

# ========== 检索配置默认值 ==========
DEFAULT_HOPS = 2
DEFAULT_TOP_K = 10

# ========== 查询时排除的"工具/操作"类关系 ==========
# 这类关系（使用工具 / 需要工具）描述"用什么工具执行某动作"，不承载故障因果语义。
# 查询"主轴轴承磨损"时若 2 跳扩展会把它们卷进来（如 更换主轴轴承 -[需要工具]-> 轴承拉拔器），
# 造成结果中混入与故障本身无关的实体关系。故默认从子图扩展中排除，保证返回的是
# 故障-原因-症状-措施-诊断 这一诊断因果链。
GRAPH_NOISE_REL_TYPES = frozenset({"使用工具", "需要工具"})
