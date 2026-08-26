# -*- coding: utf-8 -*-
"""
Neo4j知识图谱一键导入脚本
全字段建模版本：导入10种节点类型和8种关系类型
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py2neo import Graph
from config import NEO4J_CONFIG


def check_neo4j_connection():
    """检查Neo4j连接"""
    print(" 检查Neo4j连接...")
    try:
        cfg = NEO4J_CONFIG
        g = Graph(cfg["uri"], auth=(cfg["username"], cfg["password"]))
        # 测试查询
        result = g.run("RETURN 1 AS test").data()
        if result and result[0]['test'] == 1:
            print(" Neo4j连接成功!")
            print(f"  URI: {cfg['uri']}")
            print(f"  用户名: {cfg['username']}")
            return True, g
        else:
            print(" Neo4j连接测试失败")
            return False, None
    except Exception as e:
        print(f" Neo4j连接失败: {e}")
        print("\n请检查：")
        print("1. Neo4j数据库是否已启动")
        print("2. 密码是否正确")
        print("3. bolt协议端口7687是否可访问")
        return False, None


def count_nodes_and_relationships(g):
    """统计节点和关系数量"""
    print("\n 统计图谱数据...")

    # 统计节点
    node_types = ["FaultType", "Symptom", "Component", "Solution", "Check",
                  "Material", "Category", "CureWay", "Cause", "Prevent", "Parameter"]

    total_nodes = 0
    print("\n  节点统计:")
    for node_type in node_types:
        try:
            count = g.run(f"MATCH (n:{node_type}) RETURN count(n) AS count").data()[0]['count']
            if count > 0:
                print(f"    - {node_type}: {count}")
                total_nodes += count
        except:
            pass

    # 统计关系
    rel_types = ["HAS_SYMPTOM", "INVOLVES_COMPONENT", "HAS_SOLUTION",
                 "NEEDS_CHECK", "APPLIES_TO_MATERIAL", "BELONGS_TO_CATEGORY",
                 "HAS_CURE_WAY", "HAS_CAUSE", "HAS_PREVENT", "HAS_PARAMETER"]

    total_rels = 0
    print("\n  关系统计:")
    for rel_type in rel_types:
        try:
            count = g.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count").data()[0]['count']
            if count > 0:
                print(f"    - {rel_type}: {count}")
                total_rels += count
        except:
            pass

    print(f"\n  总计: {total_nodes} 个节点, {total_rels} 个关系")
    return total_nodes, total_rels


def main():
    """主函数"""
    print("=" * 80)
    print(" 机床故障诊断知识图谱 - 一键导入脚本 (全字段建模版本)")
    print("=" * 80)

    # 步骤1: 检查Neo4j连接
    success, g = check_neo4j_connection()
    if not success:
        print("\n 请先启动Neo4j数据库，然后重新运行此脚本")
        return

    # 步骤2: 确认操作
    print("\n  此操作将清除现有数据并重新构建知识图谱!")
    confirm = input("确认继续? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("操作已取消")
        return

    # 步骤3: 构建知识图谱
    print("\n 开始构建知识图谱...")
    try:
        from build_machinegraph_full import MachineFaultGraphFull
        builder = MachineFaultGraphFull()
        builder.build_full_graph()
        print(" 知识图谱构建完成!")
    except Exception as e:
        print(f" 知识图谱构建失败: {e}")
        return

    # 步骤4: 验证导入
    print("\n 验证导入结果...")
    total_nodes, total_rels = count_nodes_and_relationships(g)

    if total_nodes > 0:
        print(f"\n 导入成功! 共导入 {total_nodes} 个节点和 {total_rels} 个关系")

        # 步骤5: 测试查询
        print("\n 测试综合查询 (fault_full_info)...")
        try:
            test_query = """
            MATCH (f:FaultType {name: '刀具磨损'})
            OPTIONAL MATCH (f)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (f)-[:INVOLVES_COMPONENT]->(c:Component)
            OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
            RETURN 
                f.name AS fault,
                f.desc AS description,
                COLLECT(DISTINCT s.name) AS symptoms,
                COLLECT(DISTINCT c.name) AS components,
                COLLECT(DISTINCT sol.name) AS solutions
            """
            result = g.run(test_query).data()
            if result:
                print(" 综合查询测试通过!")
                print(f"  故障: {result[0]['fault']}")
                print(f"  症状数量: {len(result[0]['symptoms'])}")
                print(f"  部件数量: {len(result[0]['components'])}")
                print(f"  解决方案数量: {len(result[0]['solutions'])}")
        except Exception as e:
            print(f" 综合查询测试失败: {e}")

        # 测试多跳查询
        print("\n 测试多跳查询 (症状故障原因解决方案)...")
        try:
            test_query2 = """
            MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {name: '切削力增大'})
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
            OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
            RETURN 
                s.name AS symptom,
                f.name AS fault,
                COLLECT(DISTINCT c.name)[..3] AS causes,
                COLLECT(DISTINCT sol.name)[..3] AS solutions
            """
            result2 = g.run(test_query2).data()
            if result2:
                print(" 多跳查询测试通过!")
                print(f"  症状: {result2[0]['symptom']}")
                print(f"  相关故障: {result2[0]['fault']}")
        except Exception as e:
            print(f" 多跳查询测试失败: {e}")

        print("\n" + "=" * 80)
        print(" 知识图谱导入和验证完成!")
        print("=" * 80)
        print("\n接下来可以:")
        print("  1. 运行 chatbot_graph.py 启动交互式问答")
        print("  2. 访问 http://localhost:7474 使用Neo4j Browser查看图谱")
        print("  3. 运行 test_system_full.py 进行完整测试")

    else:
        print("\n 导入失败，请检查错误信息")


if __name__ == '__main__':
    main()
