# -*- coding: utf-8 -*-
"""
机床故障诊断 - 问题解析器（查询计划生成模块）

v2.0：不再生成 Neo4j Cypher，改为生成结构化查询指令。
每条指令 = {question_type, entities}，由 answer_search 在 NetworkX 图上执行。
意图类型（question_type）与字段语义完全保持历史模板一致，answer_prettify 无需改动。
"""

import re


class QuestionParser:
    """将意图识别结果转换为结构化查询计划"""

    def build_entitydict(self, args):
        """构建实体节点字典"""
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)
        return entity_dict

    def parser_main(self, res_classify):
        """解析主函数：返回 [{question_type, sql: [{question_type, entities}]}]"""
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']
        sqls = []

        for question_type in question_types:
            sql_ = {'question_type': question_type, 'sql': []}

            # 1. 故障原因查询
            if question_type == 'fault_cause':
                sql_['sql'] = self.sql_transfer('fault_cause', entity_dict.get('故障类型', []))

            # 2. 现象诊断（已知症状查故障）
            elif question_type == 'phenomenon_diagnosis':
                sql_['sql'] = self.sql_transfer('phenomenon_diagnosis', entity_dict.get('症状', []))

            # 3. 解决方法推荐
            elif question_type == 'fault_solution':
                sql_['sql'] = self.sql_transfer('fault_solution', entity_dict.get('故障类型', []))

            # 4. 部件关联故障
            elif question_type == 'component_fault':
                sql_['sql'] = self.sql_transfer('component_fault', entity_dict.get('部件', []))

            # 5. 预防措施
            elif question_type == 'fault_prevent':
                sql_['sql'] = self.sql_transfer('fault_prevent', entity_dict.get('故障类型', []))

            # 6. 参数优化（基于症状）
            elif question_type == 'parameter_optimize':
                sql_['sql'] = self.sql_transfer('parameter_optimize', entity_dict.get('症状', []))

            # 6.1 参数优化（基于故障）
            elif question_type == 'fault_parameter':
                sql_['sql'] = self.sql_transfer('fault_parameter', entity_dict.get('故障类型', []))

            # 7. 检测手段
            elif question_type == 'fault_detection':
                sql_['sql'] = self.sql_transfer('fault_detection', entity_dict.get('故障类型', []))

            # 7.1 现象检测
            elif question_type == 'phenomenon_detection':
                sql_['sql'] = self.sql_transfer('phenomenon_detection', entity_dict.get('症状', []))

            # 8. 材料相关故障
            elif question_type == 'material_fault':
                sql_['sql'] = self.sql_transfer('material_fault', entity_dict.get('材料', []))

            # 9. 故障描述
            elif question_type == 'fault_desc':
                sql_['sql'] = self.sql_transfer('fault_desc', entity_dict.get('故障类型', []))

            # 10. 故障类别查询
            elif question_type == 'fault_category':
                sql_['sql'] = self.sql_transfer('fault_category', entity_dict.get('故障类型', []))

            # 11. 修复方式查询
            elif question_type == 'fault_cure_way':
                sql_['sql'] = self.sql_transfer('fault_cure_way', entity_dict.get('故障类型', []))

            # 12. 易发情况查询
            elif question_type == 'fault_easy_get':
                sql_['sql'] = self.sql_transfer('fault_easy_get', entity_dict.get('故障类型', []))

            # 13. 修复时间查询
            elif question_type == 'fault_cure_lasttime':
                sql_['sql'] = self.sql_transfer('fault_cure_lasttime', entity_dict.get('故障类型', []))

            # 14. 修复概率查询
            elif question_type == 'fault_cured_prob':
                sql_['sql'] = self.sql_transfer('fault_cured_prob', entity_dict.get('故障类型', []))

            # 15. 原因详情查询（多跳查询）
            elif question_type == 'cause_detail':
                sql_['sql'] = self.sql_transfer('cause_detail', entity_dict.get('故障类型', []))

            # 16. 预防详情查询（多跳查询）
            elif question_type == 'prevent_detail':
                sql_['sql'] = self.sql_transfer('prevent_detail', entity_dict.get('故障类型', []))

            # 17. 综合查询：故障全信息（一次性获取所有信息）
            elif question_type == 'fault_full_info':
                sql_['sql'] = self.sql_transfer('fault_full_info', entity_dict.get('故障类型', []))

            # 18. 多跳查询：症状→故障→原因→解决方案
            elif question_type == 'symptom_to_solution':
                sql_['sql'] = self.sql_transfer('symptom_to_solution', entity_dict.get('症状', []))

            # 19. 多跳查询：部件→故障→检测方法
            elif question_type == 'component_to_check':
                sql_['sql'] = self.sql_transfer('component_to_check', entity_dict.get('部件', []))

            # 20. 多跳查询：材料→故障→预防措施
            elif question_type == 'material_to_prevent':
                sql_['sql'] = self.sql_transfer('material_to_prevent', entity_dict.get('材料', []))

            if sql_['sql']:
                sqls.append(sql_)

        return sqls

    def sql_transfer(self, question_type, entities):
        """针对不同的问题类型，生成结构化查询指令（替代 Cypher）"""
        if not entities:
            return []
        # 实体名去重（保持输入顺序）
        seen = set()
        unique_entities = []
        for entity in entities:
            e = str(entity).strip()
            if e and e not in seen:
                seen.add(e)
                unique_entities.append(e)
        return [{"question_type": question_type, "entities": unique_entities}]


if __name__ == '__main__':
    handler = QuestionParser()
    print("QuestionParser 初始化完成（NetworkX 查询计划版）")
