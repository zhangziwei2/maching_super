# -*- coding: utf-8 -*-
"""
一键重建全字段知识图谱 - 完整版
清除旧数据，创建10种节点 + 8种关系
"""

from py2neo import Graph, Node, Relationship, NodeMatcher
import os
import json
import re


class FullGraphBuilder:
    """全字段建模知识图谱构建器"""

    def __init__(self):
        # 连接Neo4j
        print("=" * 70)
        print(" 全字段知识图谱重建脚本")
        print("=" * 70)

        cfg = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "200980216"
        }

        try:
            self.g = Graph(cfg["uri"], auth=(cfg["username"], cfg["password"]))
            print(f" 连接Neo4j成功: {cfg['uri']}")
        except Exception as e:
            print(f" 连接失败: {e}")
            print("请确保Neo4j已启动，bolt端口7687可访问")
            raise

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.cur_dir, "data", "machine_fault.json")

    def clear_database(self):
        """清空数据库"""
        print("\n 清空现有数据...")
        self.g.delete_all()
        print(" 数据库已清空")

    def load_data(self):
        """加载JSON数据"""
        print("\n 加载JSON数据...")

        self.faults = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.faults.append(json.loads(line.strip()))

        print(f" 加载了 {len(self.faults)} 条故障记录")

    def create_nodes(self):
        """创建所有节点"""
        print("\n 创建节点...")

        # 收集所有唯一值
        fault_names = set()
        symptom_names = set()
        component_names = set()
        solution_names = set()
        check_names = set()
        material_names = set()
        category_names = set()
        cure_way_names = set()
        cause_names = set()
        prevent_names = set()
        parameter_names = set()

        for fault in self.faults:
            name = fault.get('name', '')
            if name:
                fault_names.add(name)

            # 症状
            for item in fault.get('symptom', []):
                symptom_names.add(item)

            # 部件
            for item in fault.get('component', []):
                component_names.add(item)

            # 解决方案
            for item in fault.get('solution', []):
                solution_names.add(item)

            # 检测方法
            for item in fault.get('check', []):
                check_names.add(item)

            # 材料
            for item in fault.get('material', []):
                material_names.add(item)

            # 类别
            for item in fault.get('category', []):
                category_names.add(item)

            # 修复方式
            for item in fault.get('cure_way', []):
                cure_way_names.add(item)

            # 原因（字符串分割）
            cause_text = fault.get('cause', '')
            if cause_text:
                causes = re.split(r'\d+\.', cause_text)
                for cause in causes:
                    cause = cause.strip()
                    if cause and len(cause) > 5:
                        cause_names.add(cause)

            # 预防（字符串分割）
            prevent_text = fault.get('prevent', '')
            if prevent_text:
                prevents = re.split(r'\d+\.', prevent_text)
                for prevent in prevents:
                    prevent = prevent.strip()
                    if prevent and len(prevent) > 5:
                        prevent_names.add(prevent)

            # 参数
            parameter_text = fault.get('parameter', '')
            if parameter_text:
                parameter_names.add(parameter_text)

        # 创建节点
        print(f"  创建 FaultType 节点: {len(fault_names)}")
        for name in fault_names:
            node = Node("FaultType", name=name)
            self.g.create(node)

        print(f"  创建 Symptom 节点: {len(symptom_names)}")
        for name in symptom_names:
            node = Node("Symptom", name=name)
            self.g.create(node)

        print(f"  创建 Component 节点: {len(component_names)}")
        for name in component_names:
            node = Node("Component", name=name)
            self.g.create(node)

        print(f"  创建 Solution 节点: {len(solution_names)}")
        for name in solution_names:
            node = Node("Solution", name=name)
            self.g.create(node)

        print(f"  创建 Check 节点: {len(check_names)}")
        for name in check_names:
            node = Node("Check", name=name)
            self.g.create(node)

        print(f"  创建 Material 节点: {len(material_names)}")
        for name in material_names:
            node = Node("Material", name=name)
            self.g.create(node)

        print(f"  创建 Category 节点: {len(category_names)}")
        for name in category_names:
            node = Node("Category", name=name)
            self.g.create(node)

        print(f"  创建 CureWay 节点: {len(cure_way_names)}")
        for name in cure_way_names:
            node = Node("CureWay", name=name)
            self.g.create(node)

        print(f"  创建 Cause 节点: {len(cause_names)}")
        for name in cause_names:
            node = Node("Cause", name=name)
            self.g.create(node)

        print(f"  创建 Prevent 节点: {len(prevent_names)}")
        for name in prevent_names:
            node = Node("Prevent", name=name)
            self.g.create(node)

        print(f"  创建 Parameter 节点: {len(parameter_names)}")
        for name in parameter_names:
            node = Node("Parameter", name=name)
            self.g.create(node)

        print(" 所有节点创建完成")

    def update_fault_properties(self):
        """更新故障节点的属性"""
        print("\n 更新故障节点属性...")

        matcher = NodeMatcher(self.g)

        for fault in self.faults:
            name = fault.get('name', '')
            if not name:
                continue

            node = matcher.match("FaultType", name=name).first()
            if node:
                node['desc'] = fault.get('desc', '')
                node['easy_get'] = fault.get('easy_get', '')
                node['cure_lasttime'] = fault.get('cure_lasttime', '')
                node['cured_prob'] = fault.get('cured_prob', '')
                self.g.push(node)

        print(" 属性更新完成")

    def create_relationships(self):
        """创建所有关系"""
        print("\n 创建关系...")

        matcher = NodeMatcher(self.g)
        rel_counts = {}

        for fault in self.faults:
            fault_name = fault.get('name', '')
            if not fault_name:
                continue

            # 获取故障节点
            fault_node = matcher.match("FaultType", name=fault_name).first()
            if not fault_node:
                continue

            # 1. HAS_SYMPTOM
            for item in fault.get('symptom', []):
                end_node = matcher.match("Symptom", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "HAS_SYMPTOM", end_node))
                    rel_counts['HAS_SYMPTOM'] = rel_counts.get('HAS_SYMPTOM', 0) + 1

            # 2. INVOLVES_COMPONENT
            for item in fault.get('component', []):
                end_node = matcher.match("Component", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "INVOLVES_COMPONENT", end_node))
                    rel_counts['INVOLVES_COMPONENT'] = rel_counts.get('INVOLVES_COMPONENT', 0) + 1

            # 3. HAS_SOLUTION
            for item in fault.get('solution', []):
                end_node = matcher.match("Solution", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "HAS_SOLUTION", end_node))
                    rel_counts['HAS_SOLUTION'] = rel_counts.get('HAS_SOLUTION', 0) + 1

            # 4. NEEDS_CHECK
            for item in fault.get('check', []):
                end_node = matcher.match("Check", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "NEEDS_CHECK", end_node))
                    rel_counts['NEEDS_CHECK'] = rel_counts.get('NEEDS_CHECK', 0) + 1

            # 5. APPLIES_TO_MATERIAL
            for item in fault.get('material', []):
                end_node = matcher.match("Material", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "APPLIES_TO_MATERIAL", end_node))
                    rel_counts['APPLIES_TO_MATERIAL'] = rel_counts.get('APPLIES_TO_MATERIAL', 0) + 1

            # 6. BELONGS_TO_CATEGORY
            for item in fault.get('category', []):
                end_node = matcher.match("Category", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "BELONGS_TO_CATEGORY", end_node))
                    rel_counts['BELONGS_TO_CATEGORY'] = rel_counts.get('BELONGS_TO_CATEGORY', 0) + 1

            # 7. HAS_CURE_WAY
            for item in fault.get('cure_way', []):
                end_node = matcher.match("CureWay", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "HAS_CURE_WAY", end_node))
                    rel_counts['HAS_CURE_WAY'] = rel_counts.get('HAS_CURE_WAY', 0) + 1

            # 8. HAS_CAUSE
            cause_text = fault.get('cause', '')
            if cause_text:
                causes = re.split(r'\d+\.', cause_text)
                for cause in causes:
                    cause = cause.strip()
                    if cause and len(cause) > 5:
                        end_node = matcher.match("Cause", name=cause).first()
                        if end_node:
                            self.g.create(Relationship(fault_node, "HAS_CAUSE", end_node))
                            rel_counts['HAS_CAUSE'] = rel_counts.get('HAS_CAUSE', 0) + 1

            # 9. HAS_PREVENT
            prevent_text = fault.get('prevent', '')
            if prevent_text:
                prevents = re.split(r'\d+\.', prevent_text)
                for prevent in prevents:
                    prevent = prevent.strip()
                    if prevent and len(prevent) > 5:
                        end_node = matcher.match("Prevent", name=prevent).first()
                        if end_node:
                            self.g.create(Relationship(fault_node, "HAS_PREVENT", end_node))
                            rel_counts['HAS_PREVENT'] = rel_counts.get('HAS_PREVENT', 0) + 1

            # 10. HAS_PARAMETER
            parameter_text = fault.get('parameter', '')
            if parameter_text:
                end_node = matcher.match("Parameter", name=parameter_text).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "HAS_PARAMETER", end_node))
                    rel_counts['HAS_PARAMETER'] = rel_counts.get('HAS_PARAMETER', 0) + 1

        print(" 关系创建完成")
        print("\n  关系统计:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"    - {rel_type}: {count}")
        print(f"  总计: {sum(rel_counts.values())} 个关系")

    def verify(self):
        """验证导入结果"""
        print("\n 验证结果...")

        node_types = ["FaultType", "Symptom", "Component", "Solution", "Check",
                      "Material", "Category", "CureWay", "Cause", "Prevent", "Parameter"]

        rel_types = ["HAS_SYMPTOM", "INVOLVES_COMPONENT", "HAS_SOLUTION",
                     "NEEDS_CHECK", "APPLIES_TO_MATERIAL", "BELONGS_TO_CATEGORY",
                     "HAS_CURE_WAY", "HAS_CAUSE", "HAS_PREVENT", "HAS_PARAMETER"]

        print("\n  节点统计:")
        total_nodes = 0
        for node_type in node_types:
            try:
                count = self.g.run(f"MATCH (n:{node_type}) RETURN count(n) AS c").data()[0]['c']
                if count > 0:
                    print(f"     {node_type}: {count}")
                    total_nodes += count
            except Exception as e:
                print(f"     {node_type}: 查询失败 {e}")

        print(f"  节点总计: {total_nodes}")

        print("\n  关系统计:")
        total_rels = 0
        for rel_type in rel_types:
            try:
                count = self.g.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").data()[0]['c']
                if count > 0:
                    print(f"     {rel_type}: {count}")
                    total_rels += count
            except Exception as e:
                print(f"     {rel_type}: 查询失败 {e}")

        print(f"  关系总计: {total_rels}")

        return total_nodes, total_rels

    def build(self):
        """执行完整构建流程"""
        self.clear_database()
        self.load_data()
        self.create_nodes()
        self.update_fault_properties()
        self.create_relationships()
        total_nodes, total_rels = self.verify()

        print("\n" + "=" * 70)
        print(" 知识图谱重建完成!")
        print(f"   节点: {total_nodes} 个")
        print(f"   关系: {total_rels} 个")
        print("=" * 70)

        print("\n 接下来:")
        print("  1. 刷新Neo4j Browser查看完整图谱")
        print("  2. 运行: python chatbot_graph.py 启动问答系统")


if __name__ == '__main__':
    builder = FullGraphBuilder()
    builder.build()
