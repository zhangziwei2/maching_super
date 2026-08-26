# -*- coding: utf-8 -*-
"""
机床故障诊断 - 问题解析器（Cypher SQL生成模块）
全字段建模版本：支持10种节点类型、8种关系类型、综合查询
"""

import re


class QuestionParser:
    """将意图识别结果转换为Neo4j Cypher查询语句 - 全字段建模版本"""

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
        """解析主函数"""
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']
        sqls = []

        for question_type in question_types:
            sql_ = {'question_type': question_type, 'sql': []}

            # 1. 故障原因查询
            if question_type == 'fault_cause':
                sql_['sql'] = self.sql_transfer('fault_cause', entity_dict.get('fault_type', []))

            # 2. 现象诊断（已知症状查故障）
            elif question_type == 'phenomenon_diagnosis':
                sql_['sql'] = self.sql_transfer('phenomenon_diagnosis', entity_dict.get('phenomenon', []))

            # 3. 解决方法推荐
            elif question_type == 'fault_solution':
                sql_['sql'] = self.sql_transfer('fault_solution', entity_dict.get('fault_type', []))

            # 4. 部件关联故障
            elif question_type == 'component_fault':
                sql_['sql'] = self.sql_transfer('component_fault', entity_dict.get('component', []))

            # 5. 预防措施
            elif question_type == 'fault_prevent':
                sql_['sql'] = self.sql_transfer('fault_prevent', entity_dict.get('fault_type', []))

            # 6. 参数优化（基于症状）
            elif question_type == 'parameter_optimize':
                sql_['sql'] = self.sql_transfer('parameter_optimize', entity_dict.get('phenomenon', []))

            # 6.1 参数优化（基于故障）
            elif question_type == 'fault_parameter':
                sql_['sql'] = self.sql_transfer('fault_parameter', entity_dict.get('fault_type', []))

            # 7. 检测手段
            elif question_type == 'fault_detection':
                sql_['sql'] = self.sql_transfer('fault_detection', entity_dict.get('fault_type', []))

            # 7.1 现象检测
            elif question_type == 'phenomenon_detection':
                sql_['sql'] = self.sql_transfer('phenomenon_detection', entity_dict.get('phenomenon', []))

            # 8. 材料相关故障
            elif question_type == 'material_fault':
                sql_['sql'] = self.sql_transfer('material_fault', entity_dict.get('material', []))

            # 9. 故障描述
            elif question_type == 'fault_desc':
                sql_['sql'] = self.sql_transfer('fault_desc', entity_dict.get('fault_type', []))

            # 10. 故障类别查询
            elif question_type == 'fault_category':
                sql_['sql'] = self.sql_transfer('fault_category', entity_dict.get('fault_type', []))

            # 11. 修复方式查询
            elif question_type == 'fault_cure_way':
                sql_['sql'] = self.sql_transfer('fault_cure_way', entity_dict.get('fault_type', []))

            # 12. 易发情况查询
            elif question_type == 'fault_easy_get':
                sql_['sql'] = self.sql_transfer('fault_easy_get', entity_dict.get('fault_type', []))

            # 13. 修复时间查询
            elif question_type == 'fault_cure_lasttime':
                sql_['sql'] = self.sql_transfer('fault_cure_lasttime', entity_dict.get('fault_type', []))

            # 14. 修复概率查询
            elif question_type == 'fault_cured_prob':
                sql_['sql'] = self.sql_transfer('fault_cured_prob', entity_dict.get('fault_type', []))

            # 15. 原因详情查询（多跳查询）
            elif question_type == 'cause_detail':
                sql_['sql'] = self.sql_transfer('cause_detail', entity_dict.get('fault_type', []))

            # 16. 预防详情查询（多跳查询）
            elif question_type == 'prevent_detail':
                sql_['sql'] = self.sql_transfer('prevent_detail', entity_dict.get('fault_type', []))

            # 17. 综合查询：故障全信息（一次性获取所有信息）
            elif question_type == 'fault_full_info':
                sql_['sql'] = self.sql_transfer('fault_full_info', entity_dict.get('fault_type', []))

            # 18. 多跳查询：症状故障原因解决方案
            elif question_type == 'symptom_to_solution':
                sql_['sql'] = self.sql_transfer('symptom_to_solution', entity_dict.get('phenomenon', []))

            # 19. 多跳查询：部件故障检测方法
            elif question_type == 'component_to_check':
                sql_['sql'] = self.sql_transfer('component_to_check', entity_dict.get('component', []))

            # 20. 多跳查询：材料故障预防措施
            elif question_type == 'material_to_prevent':
                sql_['sql'] = self.sql_transfer('material_to_prevent', entity_dict.get('material', []))

            if sql_['sql']:
                sqls.append(sql_)

        return sqls

    def sql_transfer(self, question_type, entities):
        """针对不同的问题类型，生成对应的Cypher查询"""
        if not entities:
            return []

        sql = []

        # 1. 故障原因查询
        if question_type == 'fault_cause':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_CAUSE]->(c:Cause) "
                f"RETURN f.name AS fault, c.name AS cause"
                for entity in entities
            ]

        # 2. 现象诊断（已知症状查故障）
        elif question_type == 'phenomenon_diagnosis':
            sql = [
                f"MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, s.name AS symptom, f.desc AS description"
                for entity in entities
            ]

        # 3. 解决方法推荐
        elif question_type == 'fault_solution':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_SOLUTION]->(s:Solution) "
                f"RETURN f.name AS fault, s.name AS solution"
                for entity in entities
            ]

        # 4. 部件关联故障
        elif question_type == 'component_fault':
            sql = [
                f"MATCH (f:FaultType)-[:INVOLVES_COMPONENT]->(c:Component {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, c.name AS component, f.desc AS description"
                for entity in entities
            ]

        # 5. 预防措施
        elif question_type == 'fault_prevent':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_PREVENT]->(p:Prevent) "
                f"RETURN f.name AS fault, p.name AS prevention"
                for entity in entities
            ]

        # 6. 参数优化（基于症状）
        elif question_type == 'parameter_optimize':
            sql = [
                f"MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {{name: '{entity}'}}) "
                f"MATCH (f)-[:HAS_PARAMETER]->(p:Parameter) "
                f"RETURN f.name AS fault, s.name AS symptom, p.name AS parameter"
                for entity in entities
            ]

        # 6.1 参数优化（基于故障）
        elif question_type == 'fault_parameter':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_PARAMETER]->(p:Parameter) "
                f"RETURN f.name AS fault, p.name AS parameter"
                for entity in entities
            ]

        # 7. 检测手段
        elif question_type == 'fault_detection':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:NEEDS_CHECK]->(c:Check) "
                f"RETURN f.name AS fault, c.name AS detection_method"
                for entity in entities
            ]

        # 7.1 现象检测
        elif question_type == 'phenomenon_detection':
            sql = [
                f"MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {{name: '{entity}'}}) "
                f"MATCH (f)-[:HAS_CAUSE]->(c:Cause) "
                f"RETURN f.name AS fault, s.name AS symptom, c.name AS possible_cause"
                for entity in entities
            ]

        # 8. 材料相关故障
        elif question_type == 'material_fault':
            sql = [
                f"MATCH (f:FaultType)-[:APPLIES_TO_MATERIAL]->(m:Material {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, m.name AS material, f.desc AS description"
                for entity in entities
            ]

        # 9. 故障描述
        elif question_type == 'fault_desc':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, f.desc AS description, f.easy_get AS easy_get"
                for entity in entities
            ]

        # 10. 故障类别查询
        elif question_type == 'fault_category':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:BELONGS_TO_CATEGORY]->(c:Category) "
                f"RETURN f.name AS fault, c.name AS category"
                for entity in entities
            ]

        # 11. 修复方式查询
        elif question_type == 'fault_cure_way':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_CURE_WAY]->(c:CureWay) "
                f"RETURN f.name AS fault, c.name AS cure_way"
                for entity in entities
            ]

        # 12. 易发情况查询
        elif question_type == 'fault_easy_get':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, f.easy_get AS easy_get"
                for entity in entities
            ]

        # 13. 修复时间查询
        elif question_type == 'fault_cure_lasttime':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, f.cure_lasttime AS cure_time"
                for entity in entities
            ]

        # 14. 修复概率查询
        elif question_type == 'fault_cured_prob':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}}) "
                f"RETURN f.name AS fault, f.cured_prob AS cure_probability"
                for entity in entities
            ]

        # 15. 原因详情查询（多跳查询）
        elif question_type == 'cause_detail':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_CAUSE]->(c:Cause) "
                f"RETURN f.name AS fault, c.name AS cause_detail"
                for entity in entities
            ]

        # 16. 预防详情查询（多跳查询）
        elif question_type == 'prevent_detail':
            sql = [
                f"MATCH (f:FaultType {{name: '{entity}'}})-[:HAS_PREVENT]->(p:Prevent) "
                f"RETURN f.name AS fault, p.name AS prevention_detail"
                for entity in entities
            ]

        # 17. 综合查询：故障全信息（一次性获取所有信息）
        elif question_type == 'fault_full_info':
            sql = [
                f"""
                MATCH (f:FaultType {{name: '{entity}'}})
                OPTIONAL MATCH (f)-[:HAS_SYMPTOM]->(s:Symptom)
                OPTIONAL MATCH (f)-[:INVOLVES_COMPONENT]->(c:Component)
                OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
                OPTIONAL MATCH (f)-[:NEEDS_CHECK]->(chk:Check)
                OPTIONAL MATCH (f)-[:APPLIES_TO_MATERIAL]->(m:Material)
                OPTIONAL MATCH (f)-[:BELONGS_TO_CATEGORY]->(cat:Category)
                OPTIONAL MATCH (f)-[:HAS_CURE_WAY]->(cw:CureWay)
                OPTIONAL MATCH (f)-[:HAS_CAUSE]->(cau:Cause)
                OPTIONAL MATCH (f)-[:HAS_PREVENT]->(pre:Prevent)
                OPTIONAL MATCH (f)-[:HAS_PARAMETER]->(param:Parameter)
                RETURN 
                    f.name AS fault,
                    f.desc AS description,
                    f.easy_get AS easy_get,
                    f.cure_lasttime AS cure_time,
                    f.cured_prob AS cure_probability,
                    COLLECT(DISTINCT s.name) AS symptoms,
                    COLLECT(DISTINCT c.name) AS components,
                    COLLECT(DISTINCT sol.name) AS solutions,
                    COLLECT(DISTINCT chk.name) AS checks,
                    COLLECT(DISTINCT m.name) AS materials,
                    COLLECT(DISTINCT cat.name) AS categories,
                    COLLECT(DISTINCT cw.name) AS cure_ways,
                    COLLECT(DISTINCT cau.name) AS causes,
                    COLLECT(DISTINCT pre.name) AS preventions,
                    COLLECT(DISTINCT param.name) AS parameters
                """
                for entity in entities
            ]

        # 18. 多跳查询：症状故障原因解决方案
        elif question_type == 'symptom_to_solution':
            sql = [
                f"""
                MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {{name: '{entity}'}})
                OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
                OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
                RETURN 
                    s.name AS symptom,
                    f.name AS fault,
                    COLLECT(DISTINCT c.name) AS causes,
                    COLLECT(DISTINCT sol.name) AS solutions
                """
                for entity in entities
            ]

        # 19. 多跳查询：部件故障检测方法
        elif question_type == 'component_to_check':
            sql = [
                f"""
                MATCH (f:FaultType)-[:INVOLVES_COMPONENT]->(c:Component {{name: '{entity}'}})
                OPTIONAL MATCH (f)-[:NEEDS_CHECK]->(chk:Check)
                RETURN 
                    c.name AS component,
                    f.name AS fault,
                    COLLECT(DISTINCT chk.name) AS checks
                """
                for entity in entities
            ]

        # 20. 多跳查询：材料故障预防措施
        elif question_type == 'material_to_prevent':
            sql = [
                f"""
                MATCH (f:FaultType)-[:APPLIES_TO_MATERIAL]->(m:Material {{name: '{entity}'}})
                OPTIONAL MATCH (f)-[:HAS_PREVENT]->(p:Prevent)
                RETURN 
                    m.name AS material,
                    f.name AS fault,
                    COLLECT(DISTINCT p.name) AS preventions
                """
                for entity in entities
            ]

        return sql


if __name__ == '__main__':
    handler = QuestionParser()

    # 测试用例
    test_res = {
        'args': {'刀具磨损': ['fault_type']},
        'question_types': ['fault_full_info']
    }

    result = handler.parser_main(test_res)
    for item in result:
        print(f"问题类型: {item['question_type']}")
        for sql in item['sql']:
            print(f"Cypher: {sql}\n")
