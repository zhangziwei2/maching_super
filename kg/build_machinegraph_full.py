"""
机床故障诊断知识图谱构建脚本（完整版 - JSONL 格式支持）
============================================
需求：
1. 13个主要故障类型作为一级实体（中心节点）
2. 所有详细信息（症状、原因、解决方案等）都作为独立节点，通过关系边与主节点关联
3. 所有节点和关系都在同一个知识图谱里
4. 在 Neo4j Browser 中一键查询就能看到完整图谱

数据结构：
- 主节点：FaultType（故障类型）
- 关系节点：Symptom, Cause, Solution, Check, CureWay, Component, Material, Parameter, Category, Prevention
- 所有节点都通过关系边连接到主节点

支持格式：
- JSON 数组格式：[ {...}, {...} ]
- JSONL 格式（每行一个 JSON 对象）
"""

from py2neo import Graph, Node, Relationship
import json
import os


def build_full_knowledge_graph(delete_existing=False):
    """
    构建完整的知识图谱（所有节点和关系在同一张图中）
    
    Args:
        delete_existing: 是否删除现有数据（默认 False，保留现有数据）
    """
    
    # ===== 1. 连接 Neo4j =====
    print("=" * 60)
    print("步骤1：连接 Neo4j 数据库...")
    print("=" * 60)
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "200980216"))

    # 可选：清空现有数据
    if delete_existing:
        print("\n⚠️  清空现有数据...")
        graph.delete_all()
        print("✅ 现有数据已清空")

    # ===== 2. 读取 JSON 数据 =====
    print("\n" + "=" * 60)
    print("步骤2：读取 JSON 数据...")
    print("=" * 60)

    json_path = os.path.join(os.path.dirname(__file__), "data", "machine_fault.json")
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误：文件不存在 - {json_path}")
        return
    
    # 读取 JSON 数据（优先尝试 JSON 数组，失败则尝试 JSONL 格式）
    data = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # 方式1：尝试解析为 JSON 数组
            if content.startswith("[") and content.endswith("]"):
                try:
                    data = json.loads(content)
                    print(f"✅ 以 JSON 数组格式读取，共 {len(data)} 条记录")
                except json.JSONDecodeError:
                    # 不是有效的 JSON 数组，继续尝试 JSONL
                    pass
            
            # 方式2：按 JSONL 格式读取（每行一个 JSON 对象）
            if not data:  # 如果方式1没有成功
                f.seek(0)  # 重置文件指针
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            obj = json.loads(line)
                            data.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  第 {line_num} 行 JSON 解析失败：{e}")
                            print(f"   内容：{line[:100]}...")  # 打印前100个字符
                            continue
                print(f"✅ 以 JSONL 格式读取，共 {len(data)} 条记录")
    
    except Exception as e:
        print(f"❌ 读取文件失败：{e}")
        return

    if len(data) == 0:
        print("❌ 错误：没有读取到任何数据！")
        print("请检查 JSON 文件格式是否正确")
        return

    print(f"✅ 成功读取 {len(data)} 条故障数据")

    # ===== 3. 构建知识图谱 =====
    print("\n" + "=" * 60)
    print("步骤3：构建知识图谱...")
    print("=" * 60)

    total_nodes = 0
    total_relationships = 0

    for idx, fault_data in enumerate(data, 1):
        fault_name = fault_data.get("name", f"故障_{idx}")
        print(f"\n[{idx}/{len(data)}] 处理故障类型：{fault_name}")

        # --- 3.1 创建主节点（FaultType） ---
        fault_node = Node(
            "FaultType",  # 使用 FaultType 标签（与现有数据库保持一致）
            name=fault_name,
            desc=fault_data.get("desc", ""),
            easy_get=fault_data.get("easy_get", ""),
            cure_lasttime=fault_data.get("cure_lasttime", ""),
            cured_prob=fault_data.get("cured_prob", "")
        )
        graph.create(fault_node)
        total_nodes += 1

        # --- 3.2 创建症状节点 (Symptom) ---
        for symptom_text in fault_data.get("symptom", []):
            symptom_node = Node("Symptom", text=symptom_text)
            graph.create(symptom_node)
            rel = Relationship(fault_node, "HAS_SYMPTOM", symptom_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.3 创建原因节点 (Cause) ---
        for cause_text in fault_data.get("cause", []):
            cause_node = Node("Cause", text=cause_text)
            graph.create(cause_node)
            rel = Relationship(fault_node, "HAS_CAUSE", cause_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.4 创建解决方案节点 (Solution) ---
        for solution_text in fault_data.get("solution", []):
            solution_node = Node("Solution", text=solution_text)
            graph.create(solution_node)
            rel = Relationship(fault_node, "HAS_SOLUTION", solution_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.5 创建检查方法节点 (Check) ---
        for check_text in fault_data.get("check", []):
            check_node = Node("Check", text=check_text)
            graph.create(check_node)
            rel = Relationship(fault_node, "NEEDS_CHECK", check_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.6 创建治疗方式节点 (CureWay) ---
        for cure_way_text in fault_data.get("cure_way", []):
            cure_way_node = Node("CureWay", text=cure_way_text)
            graph.create(cure_way_node)
            rel = Relationship(fault_node, "HAS_CURE_WAY", cure_way_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.7 创建部件节点 (Component) ---
        for component_text in fault_data.get("component", []):
            component_node = Node("Component", text=component_text)
            graph.create(component_node)
            rel = Relationship(fault_node, "INVOLVES_COMPONENT", component_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.8 创建材料节点 (Material) ---
        for material_text in fault_data.get("material", []):
            material_node = Node("Material", text=material_text)
            graph.create(material_node)
            rel = Relationship(fault_node, "APPLIES_TO_MATERIAL", material_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.9 创建参数节点 (Parameter) ---
        for param_data in fault_data.get("parameter", []):
            param_name = param_data.get("param_name", "")
            param_value = param_data.get("value", "")
            param_adjustment = param_data.get("adjustment", "")
            
            parameter_node = Node(
                "Parameter",
                param_name=param_name,
                value=param_value,
                adjustment=param_adjustment
            )
            graph.create(parameter_node)
            rel = Relationship(fault_node, "HAS_PARAMETER", parameter_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.10 创建类别节点 (Category) ---
        for category_text in fault_data.get("category", []):
            category_node = Node("Category", text=category_text)
            graph.create(category_node)
            rel = Relationship(fault_node, "BELONGS_TO_CATEGORY", category_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        # --- 3.11 创建预防措施节点 (Prevent) ---
        for prevent_text in fault_data.get("prevent", []):
            prevention_node = Node("Prevent", text=prevent_text)
            graph.create(prevention_node)
            rel = Relationship(fault_node, "HAS_PREVENT", prevention_node)
            graph.create(rel)
            total_nodes += 1
            total_relationships += 1

        print(f"  ✅ 已创建主节点：{fault_name}")
        print(f"  ✅ 累计节点数：{total_nodes}，累计关系数：{total_relationships}")

    # ===== 4. 构建完成 =====
    print("\n" + "=" * 60)
    print("✅ 知识图谱构建完成！")
    print("=" * 60)
    print(f"总节点数：{total_nodes}")
    print(f"总关系数：{total_relationships}")

    # ===== 5. 验证查询 =====
    print("\n" + "=" * 60)
    print("验证查询：")
    print("=" * 60)

    # 查询所有节点
    result = graph.run("MATCH (n) RETURN count(n) AS node_count")
    node_count = result.data()[0]["node_count"]
    print(f"✅ 数据库中的节点总数：{node_count}")

    # 查询所有关系
    result = graph.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
    rel_count = result.data()[0]["rel_count"]
    print(f"✅ 数据库中的关系总数：{rel_count}")

    # 查询所有标签类型
    result = graph.run("CALL db.labels() YIELD label RETURN collect(label) AS labels")
    labels = result.data()[0]["labels"]
    print(f"✅ 节点标签类型：{labels}")

    # 查询所有关系类型
    result = graph.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types")
    rel_types = result.data()[0]["types"]
    print(f"✅ 关系类型：{rel_types}")

    print("\n" + "=" * 60)
    print("在 Neo4j Browser 中执行以下查询查看完整图谱：")
    print("=" * 60)
    print("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1000")
    print("=" * 60)


if __name__ == "__main__":
    # ⚠️  注意：如果希望清空现有数据，将 delete_existing 设为 True
    build_full_knowledge_graph(delete_existing=False)
