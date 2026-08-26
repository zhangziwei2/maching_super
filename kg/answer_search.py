# -*- coding: utf-8 -*-
"""
机床故障诊断 - 答案搜索与格式化模块

v2.0：查询执行层从 py2neo/Cypher 迁移到 NetworkX 图遍历（全内嵌）。
- search_main 接收 question_parser 生成的结构化查询指令 [{question_type, entities}]
- _execute 在 NetworkX 图上执行 20 种意图查询，返回与原 Cypher 相同字段名的 dict 列表
- answer_prettify 格式化逻辑保持历史版本不变（零回归）
"""

from config import CHATBOT_CONFIG
from graph_schema import NAMESPACE_RULE


class AnswerSearcher:
    """执行结构化查询指令并格式化答案 - NetworkX 版"""

    def __init__(self):
        from graph_service import graph_service
        graph_service.ensure_ready()
        self.store = graph_service.store
        self.num_limit = CHATBOT_CONFIG["num_limit"]

    # ========== 图查询原语 ==========

    def _fault_node(self, name):
        nid = (NAMESPACE_RULE, "FaultType", name)
        if self.store.G.has_node(nid):
            return nid
        # 兜底：子串模糊匹配（对标 Yuxi 的 CONTAINS fallback），取最短匹配名
        return self._contains_node("FaultType", name)

    def _entity_node(self, entity_type, name):
        nid = (NAMESPACE_RULE, entity_type, name)
        if self.store.G.has_node(nid):
            return nid
        return self._contains_node(entity_type, name)

    def _contains_node(self, entity_type, name):
        """在指定类型节点中做子串匹配，取最短包含项（如 '出现振纹' → '存在振纹'）"""
        best = None
        for nid, attrs in self.store.G.nodes(data=True):
            if attrs.get("entity_type") != entity_type or attrs.get("namespace") != NAMESPACE_RULE:
                continue
            node_name = attrs.get("name", "")
            if name in node_name or node_name in name:
                if best is None or len(node_name) < len(self.store.G.nodes[best].get("name", "")):
                    best = nid
        return best

    def _out_names(self, nid, rel_type):
        """出边邻居名列表（按 rel_type 过滤）"""
        return [
            self.store.G.nodes[dst].get("name", "")
            for _, dst, attrs in self.store.G.out_edges(nid, data=True)
            if attrs.get("rel_type") == rel_type
        ]

    def _faults_linked(self, entity_nid, rel_type):
        """反向查询：通过指定关系指向该实体的 FaultType 节点列表"""
        result = []
        for src, _, attrs in self.store.G.in_edges(entity_nid, data=True):
            if attrs.get("rel_type") == rel_type and self.store.G.nodes[src].get("entity_type") == "FaultType":
                result.append(src)
        return result

    def _fault_attrs(self, fnid):
        return self.store.G.nodes[fnid].get("props", {})

    # ========== 执行层：20 种意图的图遍历查询 ==========

    def _execute(self, question_type, entities):
        """在 NetworkX 图上执行查询，返回字段名与历史 Cypher 对齐的 dict 列表"""
        out = []
        G = self.store.G

        for entity in entities:
            # 1. 故障原因查询
            if question_type == 'fault_cause':
                fnid = self._fault_node(entity)
                if fnid:
                    for c in self._out_names(fnid, "HAS_CAUSE"):
                        out.append({"fault": entity, "cause": c})

            # 2. 现象诊断（已知症状查故障）
            elif question_type == 'phenomenon_diagnosis':
                snid = self._entity_node("Symptom", entity)
                if snid:
                    for fnid in self._faults_linked(snid, "HAS_SYMPTOM"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"fault": fname, "symptom": entity,
                                    "description": self._fault_attrs(fnid).get("desc", "")})

            # 3. 解决方法推荐
            elif question_type == 'fault_solution':
                fnid = self._fault_node(entity)
                if fnid:
                    for s in self._out_names(fnid, "HAS_SOLUTION"):
                        out.append({"fault": entity, "solution": s})

            # 4. 部件关联故障
            elif question_type == 'component_fault':
                cnid = self._entity_node("Component", entity)
                if cnid:
                    for fnid in self._faults_linked(cnid, "INVOLVES_COMPONENT"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"fault": fname, "component": entity,
                                    "description": self._fault_attrs(fnid).get("desc", "")})

            # 5. 预防措施
            elif question_type == 'fault_prevent':
                fnid = self._fault_node(entity)
                if fnid:
                    for p in self._out_names(fnid, "HAS_PREVENT"):
                        out.append({"fault": entity, "prevention": p})

            # 6. 参数优化（基于症状）
            elif question_type == 'parameter_optimize':
                snid = self._entity_node("Symptom", entity)
                if snid:
                    for fnid in self._faults_linked(snid, "HAS_SYMPTOM"):
                        fname = G.nodes[fnid].get("name", "")
                        for p in self._out_names(fnid, "HAS_PARAMETER"):
                            out.append({"fault": fname, "symptom": entity, "parameter": p})

            # 6.1 参数优化（基于故障）
            elif question_type == 'fault_parameter':
                fnid = self._fault_node(entity)
                if fnid:
                    for _, dst, attrs in G.out_edges(fnid, data=True):
                        if attrs.get("rel_type") != "HAS_PARAMETER":
                            continue
                        props = G.nodes[dst].get("props", {})
                        out.append({"fault": entity,
                                    "parameter": props.get("param_name", G.nodes[dst].get("name", "")),
                                    "value": props.get("value", ""),
                                    "adjustment": props.get("adjustment", "")})

            # 7. 检测手段
            elif question_type == 'fault_detection':
                fnid = self._fault_node(entity)
                if fnid:
                    for c in self._out_names(fnid, "NEEDS_CHECK"):
                        out.append({"fault": entity, "detection_method": c})

            # 7.1 现象检测
            elif question_type == 'phenomenon_detection':
                snid = self._entity_node("Symptom", entity)
                if snid:
                    for fnid in self._faults_linked(snid, "HAS_SYMPTOM"):
                        fname = G.nodes[fnid].get("name", "")
                        causes = self._out_names(fnid, "HAS_CAUSE")
                        out.append({"fault": fname, "symptom": entity,
                                    "possible_cause": causes[0] if causes else ""})

            # 8. 材料相关故障
            elif question_type == 'material_fault':
                mnid = self._entity_node("Material", entity)
                if mnid:
                    for fnid in self._faults_linked(mnid, "APPLIES_TO_MATERIAL"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"fault": fname, "material": entity,
                                    "description": self._fault_attrs(fnid).get("desc", "")})

            # 9. 故障描述
            elif question_type == 'fault_desc':
                fnid = self._fault_node(entity)
                if fnid:
                    props = self._fault_attrs(fnid)
                    out.append({"fault": entity,
                                "description": props.get("desc", ""),
                                "easy_get": props.get("easy_get", "")})

            # 10. 故障类别查询
            elif question_type == 'fault_category':
                fnid = self._fault_node(entity)
                if fnid:
                    for c in self._out_names(fnid, "BELONGS_TO_CATEGORY"):
                        out.append({"fault": entity, "category": c})

            # 11. 修复方式查询
            elif question_type == 'fault_cure_way':
                fnid = self._fault_node(entity)
                if fnid:
                    for c in self._out_names(fnid, "HAS_CURE_WAY"):
                        out.append({"fault": entity, "cure_way": c})

            # 12. 易发情况查询
            elif question_type == 'fault_easy_get':
                fnid = self._fault_node(entity)
                if fnid:
                    props = self._fault_attrs(fnid)
                    out.append({"fault": entity, "easy_get": props.get("easy_get", "")})

            # 13. 修复时间查询
            elif question_type == 'fault_cure_lasttime':
                fnid = self._fault_node(entity)
                if fnid:
                    props = self._fault_attrs(fnid)
                    out.append({"fault": entity, "cure_time": props.get("cure_lasttime", "")})

            # 14. 修复概率查询
            elif question_type == 'fault_cured_prob':
                fnid = self._fault_node(entity)
                if fnid:
                    props = self._fault_attrs(fnid)
                    out.append({"fault": entity, "cure_probability": props.get("cured_prob", "")})

            # 15. 原因详情查询（多跳查询）
            elif question_type == 'cause_detail':
                fnid = self._fault_node(entity)
                if fnid:
                    for c in self._out_names(fnid, "HAS_CAUSE"):
                        out.append({"fault": entity, "cause_detail": c})

            # 16. 预防详情查询（多跳查询）
            elif question_type == 'prevent_detail':
                fnid = self._fault_node(entity)
                if fnid:
                    for p in self._out_names(fnid, "HAS_PREVENT"):
                        out.append({"fault": entity, "prevention_detail": p})

            # 17. 综合查询：故障全信息
            elif question_type == 'fault_full_info':
                fnid = self._fault_node(entity)
                if fnid:
                    props = self._fault_attrs(fnid)
                    out.append({
                        "fault": entity,
                        "description": props.get("desc", ""),
                        "easy_get": props.get("easy_get", ""),
                        "cure_time": props.get("cure_lasttime", ""),
                        "cure_probability": props.get("cured_prob", ""),
                        "symptoms": self._out_names(fnid, "HAS_SYMPTOM"),
                        "components": self._out_names(fnid, "INVOLVES_COMPONENT"),
                        "solutions": self._out_names(fnid, "HAS_SOLUTION"),
                        "checks": self._out_names(fnid, "NEEDS_CHECK"),
                        "materials": self._out_names(fnid, "APPLIES_TO_MATERIAL"),
                        "categories": self._out_names(fnid, "BELONGS_TO_CATEGORY"),
                        "cure_ways": self._out_names(fnid, "HAS_CURE_WAY"),
                        "causes": self._out_names(fnid, "HAS_CAUSE"),
                        "prevents": self._out_names(fnid, "HAS_PREVENT"),
                        "parameters": [G.nodes[dst].get("props", {}).get("param_name", G.nodes[dst].get("name", ""))
                                       for _, dst, attrs in G.out_edges(fnid, data=True)
                                       if attrs.get("rel_type") == "HAS_PARAMETER"],
                    })

            # 18. 多跳查询：症状→故障→原因→解决方案
            elif question_type == 'symptom_to_solution':
                snid = self._entity_node("Symptom", entity)
                if snid:
                    for fnid in self._faults_linked(snid, "HAS_SYMPTOM"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"symptom": entity, "fault": fname,
                                    "causes": self._out_names(fnid, "HAS_CAUSE")[:3],
                                    "solutions": self._out_names(fnid, "HAS_SOLUTION")[:3]})

            # 19. 多跳查询：部件→故障→检测方法
            elif question_type == 'component_to_check':
                cnid = self._entity_node("Component", entity)
                if cnid:
                    for fnid in self._faults_linked(cnid, "INVOLVES_COMPONENT"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"component": entity, "fault": fname,
                                    "checks": self._out_names(fnid, "NEEDS_CHECK")[:3],
                                    "causes": self._out_names(fnid, "HAS_CAUSE")[:3]})

            # 20. 多跳查询：材料→故障→预防措施
            elif question_type == 'material_to_prevent':
                mnid = self._entity_node("Material", entity)
                if mnid:
                    for fnid in self._faults_linked(mnid, "APPLIES_TO_MATERIAL"):
                        fname = G.nodes[fnid].get("name", "")
                        out.append({"material": entity, "fault": fname,
                                    "prevents": self._out_names(fnid, "HAS_PREVENT")[:3],
                                    "causes": self._out_names(fnid, "HAS_CAUSE")[:3]})

        return out

    # ========== 入口 ==========

    def search_main(self, sqls):
        """执行查询指令，并返回相应结果"""
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']
            answers = []
            for query in queries:
                answers += self._execute(question_type, query.get('entities', []))
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        """根据对应的question_type，调用相应的回复模板 - 中文版（历史版本，保持不变）"""
        final_answer = []
        if not answers:
            return ''

        # ===== 1. 故障原因查询 =====
        if question_type == 'fault_cause':
            fault_name = answers[0].get('fault', '')
            causes = [i.get('cause', '') for i in answers if i.get('cause')]
            if causes:
                final_answer = f"【{fault_name}】的可能原因包括：\n" + '\n'.join(list(set(causes))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的原因信息"

        # ===== 2. 现象诊断（已知症状查故障）=====
        elif question_type == 'phenomenon_diagnosis':
            symptom_name = answers[0].get('symptom', '')
            faults = [i.get('fault', '') for i in answers if i.get('fault')]
            if faults:
                final_answer = f"出现【{symptom_name}】现象，可能是以下故障导致的：\n" + '；'.join(list(set(faults))[:self.num_limit])
            else:
                final_answer = f"未找到与【{symptom_name}】相关的故障信息"

        # ===== 3. 解决方法推荐 =====
        elif question_type == 'fault_solution':
            fault_name = answers[0].get('fault', '')
            solutions = [i.get('solution', '') for i in answers if i.get('solution')]
            if solutions:
                final_answer = f"【{fault_name}】的解决方法包括：\n" + '\n'.join(list(set(solutions))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的解决方法"

        # ===== 4. 部件关联故障 =====
        elif question_type == 'component_fault':
            component_name = answers[0].get('component', '')
            faults = [i.get('fault', '') for i in answers if i.get('fault')]
            if faults:
                final_answer = f"【{component_name}】部件常见的故障类型包括：\n" + '；'.join(list(set(faults))[:self.num_limit])
            else:
                final_answer = f"未找到与【{component_name}】相关的故障信息"

        # ===== 5. 预防措施 =====
        elif question_type == 'fault_prevent':
            fault_name = answers[0].get('fault', '')
            prevents = [i.get('prevention', '') for i in answers if i.get('prevention')]
            if prevents:
                final_answer = f"【{fault_name}】的预防措施包括：\n" + '\n'.join(list(set(prevents))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的预防措施"

        # ===== 6. 参数优化（基于症状）=====
        elif question_type == 'parameter_optimize':
            symptom_name = answers[0].get('symptom', '')
            faults = [i.get('fault', '') for i in answers if i.get('fault')]
            parameters = [i.get('parameter', '') for i in answers if i.get('parameter')]
            if faults:
                final_answer = f"出现【{symptom_name}】时，相关故障及参数调整建议如下：\n"
                final_answer += f"可能故障：{'；'.join(list(set(faults))[:self.num_limit])}\n"
                if parameters:
                    final_answer += f"参数建议：\n" + '\n'.join(list(set(parameters))[:self.num_limit])
            else:
                final_answer = f"出现【{symptom_name}】可能涉及以下故障，建议进一步诊断：\n"
                final_answer += '；'.join(list(set(faults))[:self.num_limit])

        # ===== 6.1 参数优化（基于故障）=====
        elif question_type == 'fault_parameter':
            fault_name = answers[0].get('fault', '')
            parameters = [i.get('parameter', '') for i in answers if i.get('parameter')]
            if parameters:
                final_answer = f"【{fault_name}】的参数调整建议：\n" + '\n'.join(list(set(parameters))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的参数调整建议"

        # ===== 7. 检测手段 =====
        elif question_type == 'fault_detection':
            fault_name = answers[0].get('fault', '')
            checks = [i.get('detection_method', '') for i in answers if i.get('detection_method')]
            if checks:
                final_answer = f"【{fault_name}】的检测手段包括：\n" + '\n'.join(list(set(checks))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的检测手段"

        # ===== 7.1 现象检测 =====
        elif question_type == 'phenomenon_detection':
            symptom_name = answers[0].get('symptom', '')
            faults = [i.get('fault', '') for i in answers if i.get('fault')]
            causes = [i.get('possible_cause', '') for i in answers if i.get('possible_cause')]
            final_answer = f"出现【{symptom_name}】时，可能的故障及相关诊断方法如下：\n"
            if faults:
                final_answer += f"可能故障：{'；'.join(list(set(faults))[:self.num_limit])}\n"
            if causes:
                final_answer += f"可能原因：{'；'.join(list(set(causes))[:self.num_limit])}\n"
            final_answer += "建议结合实时传感器数据（振动、温度、切削力）进行进一步诊断。"

        # ===== 8. 材料相关故障 =====
        elif question_type == 'material_fault':
            material_name = answers[0].get('material', '')
            faults = [i.get('fault', '') for i in answers if i.get('fault')]
            if faults:
                final_answer = f"加工【{material_name}】时常见的故障类型包括：\n" + '；'.join(list(set(faults))[:self.num_limit])
            else:
                final_answer = f"未找到加工【{material_name}】时的相关故障信息"

        # ===== 9. 故障描述 =====
        elif question_type == 'fault_desc':
            fault_name = answers[0].get('fault', '')
            desc = answers[0].get('description', '')
            easy_get = answers[0].get('easy_get', '')
            parts = [f"【{fault_name}】简介：{desc if desc else '暂无'}"]
            if easy_get:
                parts.append(f"易发情况：{easy_get}")
            final_answer = '\n'.join(parts)

        # ===== 10. 故障类别查询 =====
        elif question_type == 'fault_category':
            fault_name = answers[0].get('fault', '')
            categories = [i.get('category', '') for i in answers if i.get('category')]
            if categories:
                final_answer = f"【{fault_name}】属于以下类别：\n" + '；'.join(list(set(categories))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的类别信息"

        # ===== 11. 修复方式查询 =====
        elif question_type == 'fault_cure_way':
            fault_name = answers[0].get('fault', '')
            cure_ways = [i.get('cure_way', '') for i in answers if i.get('cure_way')]
            if cure_ways:
                final_answer = f"【{fault_name}】的修复方式包括：\n" + '\n'.join(list(set(cure_ways))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的修复方式"

        # ===== 12. 易发情况查询 =====
        elif question_type == 'fault_easy_get':
            fault_name = answers[0].get('fault', '')
            easy_get = answers[0].get('easy_get', '')
            if easy_get:
                final_answer = f"【{fault_name}】的易发情况：\n{easy_get}"
            else:
                final_answer = f"未找到【{fault_name}】的易发情况信息"

        # ===== 13. 修复时间查询 =====
        elif question_type == 'fault_cure_lasttime':
            fault_name = answers[0].get('fault', '')
            cure_time = answers[0].get('cure_time', '')
            if cure_time:
                final_answer = f"【{fault_name}】的修复时间：\n{cure_time}"
            else:
                final_answer = f"未找到【{fault_name}】的修复时间信息"

        # ===== 14. 修复概率查询 =====
        elif question_type == 'fault_cured_prob':
            fault_name = answers[0].get('fault', '')
            cure_prob = answers[0].get('cure_probability', '')
            if cure_prob:
                final_answer = f"【{fault_name}】的修复概率：\n{cure_prob}"
            else:
                final_answer = f"未找到【{fault_name}】的修复概率信息"

        # ===== 15. 原因详情查询（多跳查询）=====
        elif question_type == 'cause_detail':
            fault_name = answers[0].get('fault', '')
            causes = [i.get('cause_detail', '') for i in answers if i.get('cause_detail')]
            if causes:
                final_answer = f"【{fault_name}】的详细原因：\n" + '\n'.join(list(set(causes))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的详细原因信息"

        # ===== 16. 预防详情查询（多跳查询）=====
        elif question_type == 'prevent_detail':
            fault_name = answers[0].get('fault', '')
            prevents = [i.get('prevention_detail', '') for i in answers if i.get('prevention_detail')]
            if prevents:
                final_answer = f"【{fault_name}】的详细预防措施：\n" + '\n'.join(list(set(prevents))[:self.num_limit])
            else:
                final_answer = f"未找到【{fault_name}】的详细预防措施"

        # ===== 17. 综合查询：故障全信息（一次性获取所有信息）=====
        elif question_type == 'fault_full_info':
            fault_name = answers[0].get('fault', '')
            desc = answers[0].get('description', '')
            easy_get = answers[0].get('easy_get', '')
            cure_time = answers[0].get('cure_time', '')
            cure_prob = answers[0].get('cure_probability', '')

            symptoms = answers[0].get('symptoms', [])
            components = answers[0].get('components', [])
            solutions = answers[0].get('solutions', [])
            checks = answers[0].get('checks', [])
            materials = answers[0].get('materials', [])
            categories = answers[0].get('categories', [])
            cure_ways = answers[0].get('cure_ways', [])
            causes = answers[0].get('causes', [])
            prevents = answers[0].get('prevents', [])
            parameters = answers[0].get('parameters', [])

            parts = [f"【{fault_name}】完整信息：\n"]

            if desc:
                parts.append(f" 描述：{desc}")
            if categories:
                parts.append(f"  类别：{'，'.join(categories)}")
            if symptoms:
                parts.append(f" 症状：{'，'.join(symptoms)}")
            if causes:
                parts.append(f" 原因：{'；'.join(causes)}")
            if prevents:
                parts.append(f" 预防：{'；'.join(prevents)}")
            if parameters:
                parts.append(f"  参数建议：{'；'.join(parameters)}")
            if components:
                parts.append(f" 涉及部件：{'，'.join(components)}")
            if solutions:
                parts.append(f" 解决方法：{'，'.join(solutions)}")
            if cure_ways:
                parts.append(f" 修复方式：{'，'.join(cure_ways)}")
            if checks:
                parts.append(f" 检测方法：{'，'.join(checks)}")
            if materials:
                parts.append(f" 适用材料：{'，'.join(materials)}")
            if easy_get:
                parts.append(f" 易发情况：{easy_get}")
            if cure_time:
                parts.append(f"  修复时间：{cure_time}")
            if cure_prob:
                parts.append(f" 修复概率：{cure_prob}")

            final_answer = '\n'.join(parts)

        # ===== 18. 多跳查询：症状故障原因解决方案 =====
        elif question_type == 'symptom_to_solution':
            symptom_name = answers[0].get('symptom', '')
            faults = list(set([i.get('fault', '') for i in answers if i.get('fault')]))
            causes = list(set([c for i in answers for c in (i.get('causes', []) or []) if c]))
            solutions = list(set([s for i in answers for s in (i.get('solutions', []) or []) if s]))

            final_answer = f"出现【{symptom_name}】现象的诊断与解决方案：\n"
            if faults:
                final_answer += f"可能故障：{'；'.join(faults[:self.num_limit])}\n"
            if causes:
                final_answer += f"可能原因：{'；'.join(causes[:self.num_limit])}\n"
            if solutions:
                final_answer += f"推荐解决方案：{'；'.join(solutions[:self.num_limit])}"

        # ===== 19. 多跳查询：部件故障检测方法 =====
        elif question_type == 'component_to_check':
            component_name = answers[0].get('component', '')
            faults = list(set([i.get('fault', '') for i in answers if i.get('fault')]))
            checks = list(set([c for i in answers for c in (i.get('checks', []) or []) if c]))

            final_answer = f"【{component_name}】部件相关故障及检测方法：\n"
            if faults:
                final_answer += f"相关故障：{'；'.join(faults[:self.num_limit])}\n"
            if checks:
                final_answer += f"推荐检测方法：{'；'.join(checks[:self.num_limit])}"

        # ===== 20. 多跳查询：材料故障预防措施 =====
        elif question_type == 'material_to_prevent':
            material_name = answers[0].get('material', '')
            faults = list(set([i.get('fault', '') for i in answers if i.get('fault')]))
            prevents = list(set([p for i in answers for p in (i.get('prevents', []) or []) if p]))

            final_answer = f"加工【{material_name}】时的故障及预防措施：\n"
            if faults:
                final_answer += f"常见故障：{'；'.join(faults[:self.num_limit])}\n"
            if prevents:
                final_answer += f"预防措施：{'；'.join(prevents[:self.num_limit])}"

        return final_answer


if __name__ == '__main__':
    searcher = AnswerSearcher()
    print("AnswerSearcher 初始化成功（NetworkX 版）")
