# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱构建模块
基于医疗知识图谱项目重构，适配机床故障诊断领域

节点类型: FaultType, Component, Phenomenon, Cause, Solution, Detection, Material, Parameter
关系类型: has_symptom, has_cause, belongs_to, has_solution, need_check, has_prevent, has_parameter, applies_to_material
"""

import os
import json
from py2neo import Graph, Node


class MachineFaultGraph:

    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="200980216"):
        """初始化 Neo4j 连接"""
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'machine_fault.json')
        self.g = Graph(uri, auth=(username, password))
        print(" Neo4j 连接成功")


    def read_nodes(self):
        """读取数据文件，提取各类节点和关系"""
        fault_types = []      # 故障类型
        components = []       # 部件
        phenomena = []        # 现象
        causes = []           # 原因节点列表
        solutions = []        # 解决方法
        detections = []       # 检测方法
        materials = []        # 加工材料
        prevents = []         # 预防节点列表
        parameters = []       # 加工参数
        cure_ways = []       # 修复方式节点列表
        categories = []       # 类别节点列表

        fault_infos = []      # 故障详细信息

        # 关系列表
        rels_symptom = []         # 故障-现象
        rels_component = []       # 故障-部件
        rels_solution = []        # 故障-解决方法
        rels_detection = []       # 故障-检测方法
        rels_material = []        # 故障-适用材料
        rels_acompany = []        # 故障并发关系
        rels_category = []        # 故障-分类
        rels_cure_way = []       # 故障-修复方式
        rels_cause = []           # 故障-原因
        rels_prevent = []         # 故障-预防
        rels_parameter = []       # 故障-参数建议

        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                print(f"Processing {count}...")
                data_json = json.loads(line)

                fault = data_json['name']
                fault_dict = {
                    'name': fault,
                    'desc': data_json.get('desc', ''),
                    'cure_way': data_json.get('cure_way', []),
                    'easy_get': data_json.get('easy_get', ''),
                    'parameter': data_json.get('parameter', ''),
                    'cure_lasttime': data_json.get('cure_lasttime', ''),
                    'cured_prob': data_json.get('cured_prob', '')
                }
                fault_types.append(fault)
                fault_infos.append(fault_dict)

                # 现象关系
                if 'symptom' in data_json:
                    for symptom in data_json['symptom']:
                        phenomena.append(symptom)
                        rels_symptom.append([fault, symptom])

                # 部件关系
                if 'component' in data_json:
                    for comp in data_json['component']:
                        components.append(comp)
                        rels_component.append([fault, comp])

                # 解决方法关系
                if 'solution' in data_json:
                    for sol in data_json['solution']:
                        solutions.append(sol)
                        rels_solution.append([fault, sol])

                # 检测方法关系
                if 'check' in data_json:
                    for check in data_json['check']:
                        detections.append(check)
                        rels_detection.append([fault, check])

                # 材料关系
                if 'material' in data_json:
                    for mat in data_json['material']:
                        materials.append(mat)
                        rels_material.append([fault, mat])

                # 分类
                if 'category' in data_json:
                    for cat in data_json['category']:
                        rels_category.append([fault, cat])

                # 原因节点和关系处理
                cause_list = data_json.get('cause', [])
                if isinstance(cause_list, list):
                    for cause in cause_list:
                        cause = str(cause).strip()
                        if cause and len(cause) > 2 and cause not in causes:
                            causes.append(cause)
                            rels_cause.append([fault, cause])
                elif isinstance(cause_list, str) and cause_list.strip():
                    import re
                    for cause in re.split(r'\d+\.', cause_list):
                        cause = cause.strip()
                        if cause and len(cause) > 2 and cause not in causes:
                            causes.append(cause)
                            rels_cause.append([fault, cause])

                # 预防节点和关系处理
                prevent_list = data_json.get('prevent', [])
                if isinstance(prevent_list, list):
                    for prevent in prevent_list:
                        prevent = str(prevent).strip()
                        if prevent and len(prevent) > 2 and prevent not in prevents:
                            prevents.append(prevent)
                            rels_prevent.append([fault, prevent])
                elif isinstance(prevent_list, str) and prevent_list.strip():
                    import re
                    for prevent in re.split(r'\d+\.', prevent_list):
                        prevent = prevent.strip()
                        if prevent and len(prevent) > 2 and prevent not in prevents:
                            prevents.append(prevent)
                            rels_prevent.append([fault, prevent])

                # 参数节点和关系处理
                param_list = data_json.get('parameter', [])
                if isinstance(param_list, list):
                    for param in param_list:
                        if not isinstance(param, dict):
                            continue
                        param_name = param.get('param_name', '').strip()
                        param_value = param.get('value', '').strip()
                        param_adjust = param.get('adjustment', '').strip()
                        if not param_name:
                            continue
                        param_node_name = f"{param_name}_{param_value}" if param_value else param_name
                        # 去重添加参数节点
                        if not any(p['name'] == param_node_name for p in parameters):
                            parameters.append({
                                'name': param_node_name,
                                'param_name': param_name,
                                'param_value': param_value,
                                'param_adjustment': param_adjust
                            })
                        rels_parameter.append([fault, param_node_name])
                elif isinstance(param_list, str) and param_list.strip():
                    param_name = param_list.strip()
                    if not any(p['name'] == param_name for p in parameters):
                        parameters.append({'name': param_name})
                    rels_parameter.append([fault, param_name])

                # 修复方式节点和关系处理
                cure_way_list = data_json.get('cure_way', [])
                if isinstance(cure_way_list, list):
                    for cure_way in cure_way_list:
                        cure_way = str(cure_way).strip()
                        if cure_way and cure_way not in cure_ways:
                            cure_ways.append(cure_way)
                            rels_cure_way.append([fault, cure_way])

        return (set(fault_types), set(components), set(phenomena), set(causes),
                set(solutions), set(detections), set(materials), set(prevents),
                parameters, set(cure_ways), set(categories),
                fault_infos, rels_symptom, rels_component, rels_solution,
                rels_detection, rels_material, rels_category, rels_cure_way,
                rels_cause, rels_prevent, rels_parameter)

    def create_node(self, label, nodes):
        """批量创建节点"""
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            if count % 50 == 0:
                print(f"Created {count}/{len(nodes)} {label} nodes")
        return

    def create_parameter_nodes(self, parameters):
        """创建参数节点（带多属性）"""
        count = 0
        for param in parameters:
            node = Node(
                "参数",
                name=param['name'],
                param_name=param['param_name'],
                param_value=param['param_value'],
                param_adjustment=param['param_adjustment']
            )
            self.g.create(node)
            count += 1
            if count % 20 == 0:
                print(f"Created {count}/{len(parameters)} Parameter nodes")
        print(f"Total Parameter nodes created: {count}")
        return

    def create_cause_nodes(self, causes):
        """创建原因节点"""
        count = 0
        for cause in set(causes):
            node = Node("原因", name=cause)
            self.g.create(node)
            count += 1
        print(f"Total 原因 nodes created: {count}")
        return

    def create_prevent_nodes(self, prevents):
        """创建预防节点"""
        count = 0
        for prevent in set(prevents):
            node = Node("预防", name=prevent)
            self.g.create(node)
            count += 1
        print(f"Total 预防 nodes created: {count}")
        return

    def create_cure_way_nodes(self, cure_ways):
        """创建修复方式节点"""
        count = 0
        for cure_way in set(cure_ways):
            node = Node("修复方式", name=cure_way)
            self.g.create(node)
            count += 1
        print(f"Total 修复方式 nodes created: {count}")
        return

    def create_category_nodes(self, categories):
        """创建类别节点"""
        count = 0
        for category in set(categories):
            node = Node("类别", name=category)
            self.g.create(node)
            count += 1
        print(f"Total 类别 nodes created: {count}")
        return

    def create_fault_nodes(self, fault_infos):
        """创建故障类型中心节点（含详细属性）"""
        count = 0
        for fault_dict in fault_infos:
            node = Node(
                "故障类型",
                name=fault_dict['name'],
                描述=fault_dict['desc'],
                修复方式=','.join(fault_dict['cure_way']) if fault_dict['cure_way'] else '',
                易发情况=fault_dict['easy_get'],
                修复时间=fault_dict['cure_lasttime'],
                修复概率=fault_dict['cured_prob']
            )
            self.g.create(node)
            count += 1
            if count % 10 == 0:
                print(f"Created {count}/{len(fault_infos)} FaultType nodes")
        return

    def create_graphnodes(self):
        """创建所有节点"""
        (FaultTypes, Components, Phenomena, Causes,
         Solutions, Detections, Materials, Prevents,
         Parameters, CureWays, Categories, fault_infos,
         rels_symptom, rels_component, rels_solution, rels_detection,
         rels_material, rels_category, rels_cure_way,
         rels_cause, rels_prevent, rels_parameter) = self.read_nodes()

        print(f"\nFaultTypes: {len(FaultTypes)}")
        print(f"Components: {len(Components)}")
        print(f"Phenomena: {len(Phenomena)}")
        print(f"Sauses: {len(Causes)}")
        print(f"Solutions: {len(Solutions)}")
        print(f"Detections: {len(Detections)}")
        print(f"Materials: {len(Materials)}")
        print(f"Prevents: {len(Prevents)}")
        print(f"Parameters: {len(Parameters)}")
        print(f"CureWays: {len(CureWays)}")
        print(f"Categories: {len(Categories)}")

        self.create_fault_nodes(fault_infos)
        self.create_node('部件', Components)
        self.create_node('症状', Phenomena)
        self.create_node('解决方案', Solutions)
        self.create_node('检查', Detections)
        self.create_node('材料', Materials)
        self.create_cause_nodes(Causes)
        self.create_prevent_nodes(Prevents)
        self.create_cure_way_nodes(CureWays)
        self.create_category_nodes(Categories)
        self.create_parameter_nodes(Parameters)
        print("All nodes created!")
        return

    def create_graphrels(self):
        """创建所有关系"""
        (FaultTypes, Components, Phenomena, Causes,
         Solutions, Detections, Materials, Prevents,
         Parameters, CureWays, Categories, fault_infos,
         rels_symptom, rels_component, rels_solution, rels_detection,
         rels_material, rels_category, rels_cure_way,
         rels_cause, rels_prevent, rels_parameter) = self.read_nodes()

        self.create_relationship('故障类型', '症状', rels_symptom, '故障', '故障')
        self.create_relationship('故障类型', '部件', rels_component, '涉及部件', '涉及部件')
        self.create_relationship('故障类型', '解决方案', rels_solution, '有解决方案', '有解决方案')
        self.create_relationship('故障类型', '检查', rels_detection, '需要检查', '需要检查')
        self.create_relationship('故障类型', '材料', rels_material, '适用于材料', '适用于材料')
        self.create_relationship('故障类型', '原因', rels_cause, '原因', '原因')
        self.create_relationship('故障类型', '预防', rels_prevent, '有预防措施', '有预防措施')
        self.create_relationship('故障类型', '修复方式', rels_cure_way, '有修复方式', '有修复方式')
        self.create_relationship('故障类型', '类别', rels_category, '属于类别', '属于类别')
        self.create_relationship('故障类型', '参数', rels_parameter, '有参数建议', '有参数建议')
        print("All relationships created!")
        return

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        """创建实体关系边"""
        count = 0
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all_count = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = ("match(p:%s),(q:%s) where p.name='%s' and q.name='%s' "
                     "create (p)-[rel:%s{name:'%s'}]->(q)") % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                if count % 20 == 0:
                    print(f"{rel_type}: {count}/{all_count}")
            except Exception as e:
                print(f"Error creating relationship: {e}")
        return

    def export_data(self):
        """导出节点数据到dict目录"""
        (FaultTypes, Components, Phenomena, Causes,
         Solutions, Detections, Materials, Prevents,
         Parameters, CureWays, Categories, fault_infos,
         rels_symptom, rels_component, rels_solution, rels_detection,
         rels_material, rels_category, rels_cure_way,
         rels_cause, rels_prevent, rels_parameter) = self.read_nodes()

        dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict')
        os.makedirs(dict_dir, exist_ok=True)

        with open(os.path.join(dict_dir, 'fault_type.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(FaultTypes)))
        with open(os.path.join(dict_dir, 'component.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Components)))
        with open(os.path.join(dict_dir, 'phenomenon.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Phenomena)))
        with open(os.path.join(dict_dir, 'cause.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Causes)))
        with open(os.path.join(dict_dir, 'solution.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Solutions)))
        with open(os.path.join(dict_dir, 'detection.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Detections)))
        with open(os.path.join(dict_dir, 'material.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Materials)))
        with open(os.path.join(dict_dir, 'prevent.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Prevents)))
        # 导出参数实体（取param_name去重）
        param_names = sorted(list(set(p['param_name'] for p in Parameters)))
        with open(os.path.join(dict_dir, 'parameter.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(param_names))
        with open(os.path.join(dict_dir, 'cure_way.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(CureWays)))
        with open(os.path.join(dict_dir, 'category.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(Categories)))
        print("Data exported to dict/ directory")
        return


# if __name__ == '__main__':
#     # 请修改密码
#     handler = MachineFaultGraph(password="your_password_here")
#     # handler.export_data()
#     handler.create_graphnodes()
#     handler.create_graphrels()
if __name__ == '__main__':
    # 使用正确的 bolt 协议
    handler = MachineFaultGraph(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="200980216"  # 改成你的真实密码
    )
    # handler.export_data()
    handler.create_graphnodes()
    handler.create_graphrels()
