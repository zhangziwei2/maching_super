# -*- coding: utf-8 -*-
"""
机床故障诊断 - 答案搜索与格式化模块
全字段建模版本：支持10种节点类型、8种关系类型、综合查询
"""

from py2neo import Graph
from config import NEO4J_CONFIG, CHATBOT_CONFIG


class AnswerSearcher:
    """执行Cypher查询并格式化答案 - 全字段建模版本"""

    def __init__(self):
        cfg = NEO4J_CONFIG
        self.g = Graph(cfg["uri"], auth=(cfg["username"], cfg["password"]))
        self.num_limit = CHATBOT_CONFIG["num_limit"]

    def search_main(self, sqls):
        """执行cypher查询，并返回相应结果"""
        final_answers = []
        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']
            answers = []
            for query in queries:
                try:
                    ress = self.g.run(query).data()
                    answers += ress
                except Exception as e:
                    print(f"Query error: {e}")
                    continue
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        """根据对应的question_type，调用相应的回复模板"""
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
            prevents = answers[0].get('preventions', [])
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
            prevents = list(set([p for i in answers for p in (i.get('preventions', []) or []) if p]))

            final_answer = f"加工【{material_name}】时的故障及预防措施：\n"
            if faults:
                final_answer += f"常见故障：{'；'.join(faults[:self.num_limit])}\n"
            if prevents:
                final_answer += f"预防措施：{'；'.join(prevents[:self.num_limit])}"

        return final_answer


if __name__ == '__main__':
    searcher = AnswerSearcher()
    print("AnswerSearcher initialized successfully!")
