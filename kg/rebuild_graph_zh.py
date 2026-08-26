# -*- coding: utf-8 -*-
"""
一键重建全字段知识图谱 - 中文标签版
使用中文节点标签和中文关系名
"""

from py2neo import Graph, Node, Relationship, NodeMatcher
import os
import json
import re


class FullGraphBuilder:
    """全字段建模知识图谱构建器 - 中文版"""

    def __init__(self):
        # 连接Neo4j
        print("=" * 70)
        print(" 全字段知识图谱重建脚本 - 中文标签版")
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
        """创建所有节点 - 使用中文标签"""
        print("\n 创建节点（中文标签）...")

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

            # 原因（数组）
            for item in fault.get('cause', []):
                if item and len(item) > 2:
                    cause_names.add(item)

            # 预防（数组）
            for item in fault.get('prevent', []):
                if item and len(item) > 2:
                    prevent_names.add(item)

            # 参数（结构化数组）
            for item in fault.get('parameter', []):
                param_name = item.get('param_name', '')
                if param_name:
                    parameter_names.add(param_name)

        # 创建节点 - 使用中文标签
        print(f"  创建 故障类型 节点: {len(fault_names)}")
        for name in fault_names:
            node = Node("故障类型", name=name)
            self.g.create(node)

        print(f"  创建 症状 节点: {len(symptom_names)}")
        for name in symptom_names:
            node = Node("症状", name=name)
            self.g.create(node)

        print(f"  创建 部件 节点: {len(component_names)}")
        for name in component_names:
            node = Node("部件", name=name)
            self.g.create(node)

        print(f"  创建 解决方案 节点: {len(solution_names)}")
        for name in solution_names:
            node = Node("解决方案", name=name)
            self.g.create(node)

        print(f"  创建 检查 节点: {len(check_names)}")
        for name in check_names:
            node = Node("检查", name=name)
            self.g.create(node)

        print(f"  创建 材料 节点: {len(material_names)}")
        for name in material_names:
            node = Node("材料", name=name)
            self.g.create(node)

        print(f"  创建 类别 节点: {len(category_names)}")
        for name in category_names:
            node = Node("类别", name=name)
            self.g.create(node)

        print(f"  创建 修复方式 节点: {len(cure_way_names)}")
        for name in cure_way_names:
            node = Node("修复方式", name=name)
            self.g.create(node)

        print(f"  创建 原因 节点: {len(cause_names)}")
        for name in cause_names:
            node = Node("原因", name=name)
            self.g.create(node)

        print(f"  创建 预防 节点: {len(prevent_names)}")
        for name in prevent_names:
            node = Node("预防", name=name)
            self.g.create(node)

        print(f"  创建 参数 节点: {len(parameter_names)}")
        for name in parameter_names:
            node = Node("参数", name=name)
            self.g.create(node)

        print(" 所有节点创建完成")

        # 保存数据供后续使用
        self.node_counts = {
            "故障类型": len(fault_names),
            "症状": len(symptom_names),
            "部件": len(component_names),
            "解决方案": len(solution_names),
            "检查": len(check_names),
            "材料": len(material_names),
            "类别": len(category_names),
            "修复方式": len(cure_way_names),
            "原因": len(cause_names),
            "预防": len(prevent_names),
            "参数": len(parameter_names)
        }

    def update_fault_properties(self):
        """更新故障节点的属性"""
        print("\n 更新故障节点属性...")

        matcher = NodeMatcher(self.g)

        for fault in self.faults:
            name = fault.get('name', '')
            if not name:
                continue

            node = matcher.match("故障类型", name=name).first()
            if node:
                node['描述'] = fault.get('desc', '')
                node['易发情况'] = fault.get('easy_get', '')
                node['修复时间'] = fault.get('cure_lasttime', '')
                node['修复概率'] = fault.get('cured_prob', '')
                self.g.push(node)

        print(" 属性更新完成")

    def create_relationships(self):
        """创建所有关系 - 使用中文关系名"""
        print("\n 创建关系（中文关系名）...")

        matcher = NodeMatcher(self.g)
        rel_counts = {}

        for fault in self.faults:
            fault_name = fault.get('name', '')
            if not fault_name:
                continue

            # 获取故障节点
            fault_node = matcher.match("故障类型", name=fault_name).first()
            if not fault_node:
                continue

            # 1. 故障（故障类型 -> 症状）
            for item in fault.get('symptom', []):
                end_node = matcher.match("症状", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "故障", end_node))
                    rel_counts['故障'] = rel_counts.get('故障', 0) + 1

            # 2. 涉及部件（故障类型 -> 部件）
            for item in fault.get('component', []):
                end_node = matcher.match("部件", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "涉及部件", end_node))
                    rel_counts['涉及部件'] = rel_counts.get('涉及部件', 0) + 1

            # 3. 有解决方案（故障类型 -> 解决方案）
            for item in fault.get('solution', []):
                end_node = matcher.match("解决方案", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "有解决方案", end_node))
                    rel_counts['有解决方案'] = rel_counts.get('有解决方案', 0) + 1

            # 4. 需要检查（故障类型 -> 检查）
            for item in fault.get('check', []):
                end_node = matcher.match("检查", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "需要检查", end_node))
                    rel_counts['需要检查'] = rel_counts.get('需要检查', 0) + 1

            # 5. 适用于材料（故障类型 -> 材料）
            for item in fault.get('material', []):
                end_node = matcher.match("材料", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "适用于材料", end_node))
                    rel_counts['适用于材料'] = rel_counts.get('适用于材料', 0) + 1

            # 6. 属于类别（故障类型 -> 类别）
            for item in fault.get('category', []):
                end_node = matcher.match("类别", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "属于类别", end_node))
                    rel_counts['属于类别'] = rel_counts.get('属于类别', 0) + 1

            # 7. 有修复方式（故障类型 -> 修复方式）
            for item in fault.get('cure_way', []):
                end_node = matcher.match("修复方式", name=item).first()
                if end_node:
                    self.g.create(Relationship(fault_node, "有修复方式", end_node))
                    rel_counts['有修复方式'] = rel_counts.get('有修复方式', 0) + 1

            # 8. 原因（故障类型 -> 原因）
            for item in fault.get('cause', []):
                if item and len(item) > 2:
                    end_node = matcher.match("原因", name=item).first()
                    if end_node:
                        self.g.create(Relationship(fault_node, "原因", end_node))
                        rel_counts['原因'] = rel_counts.get('原因', 0) + 1

            # 9. 有预防措施（故障类型 -> 预防）
            for item in fault.get('prevent', []):
                if item and len(item) > 2:
                    end_node = matcher.match("预防", name=item).first()
                    if end_node:
                        self.g.create(Relationship(fault_node, "有预防措施", end_node))
                        rel_counts['有预防措施'] = rel_counts.get('有预防措施', 0) + 1

            # 10. 有参数建议（故障类型 -> 参数）
            for item in fault.get('parameter', []):
                param_name = item.get('param_name', '')
                if param_name:
                    end_node = matcher.match("参数", name=param_name).first()
                    if end_node:
                        # 给关系添加属性
                        rel = Relationship(fault_node, "有参数建议", end_node)
                        rel['值'] = item.get('value', '')
                        rel['调整建议'] = item.get('adjustment', '')
                        self.g.create(rel)
                        rel_counts['有参数建议'] = rel_counts.get('有参数建议', 0) + 1

        print(" 关系创建完成")
        print("\n  关系统计:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"    - {rel_type}: {count}")
        print(f"  总计: {sum(rel_counts.values())} 个关系")

    def verify(self):
        """验证导入结果"""
        print("\n 验证结果...")

        node_types = ["故障类型", "症状", "部件", "解决方案", "检查",
                      "材料", "类别", "修复方式", "原因", "预防", "参数"]

        rel_types = ["故障", "涉及部件", "有解决方案",
                     "需要检查", "适用于材料", "属于类别",
                     "有修复方式", "原因", "有预防措施", "有参数建议"]

        print("\n  节点统计:")
        total_nodes = 0
        for node_type in node_types:
            try:
                count = self.g.run(f"MATCH (n:`{node_type}`) RETURN count(n) AS c").data()[0]['c']
                if count > 0:
                    print(f"     {node_type}: {count}")
                    total_nodes += count
            except Exception as e:
                print(f"     {node_type}: 查询失败 {e}")

        print(f"\n  节点总计: {total_nodes}")

        print("\n  关系统计:")
        total_rels = 0
        for rel_type in rel_types:
            try:
                count = self.g.run(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c").data()[0]['c']
                if count > 0:
                    print(f"     {rel_type}: {count}")
                    total_rels += count
            except Exception as e:
                print(f"     {rel_type}: 查询失败 {e}")

        print(f"\n  关系总计: {total_rels}")

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
        print("  3. 查看中文节点标签和关系名")


if __name__ == '__main__':
    builder = FullGraphBuilder()
    builder.build()
