"""
机床故障诊断问答系统 - 简化版（直接查询知识图谱）
==============================================
功能：
1. 直接查询 Neo4j 知识图谱
2. 无需复杂的意图识别
3. 支持多种查询类型
4. 可以直接集成到 FastAPI 后端

使用方法：
    python simple_chatbot.py  # 交互式测试
    
或者在代码中调用：
    from simple_chatbot import SimpleChatBot
    bot = SimpleChatBot()
    answer = bot.ask("刀具磨损是什么原因？")
"""

from py2neo import Graph
import re


class SimpleChatBot:
    """简化版机床故障诊断问答机器人"""
    
    def __init__(self, neo4j_password="200980216"):
        """
        初始化问答机器人
        
        Args:
            neo4j_password: Neo4j 密码
        """
        print("正在连接 Neo4j 数据库...")
        self.graph = Graph(
            uri="bolt://localhost:7687",
            auth=("neo4j", neo4j_password)
        )
        print("✅ Neo4j 连接成功！")
        print(f"   数据库中有 {self._count_nodes()} 个节点，{self._count_relationships()} 个关系")
        print()
    
    def _count_nodes(self):
        """统计节点数量"""
        result = self.graph.run("MATCH (n) RETURN count(n) AS count")
        return result.data()[0]["count"]
    
    def _count_relationships(self):
        """统计关系数量"""
        result = self.graph.run("MATCH ()-[r]->() RETURN count(r) AS count")
        return result.data()[0]["count"]
    
    def ask(self, question):
        """
        问答主函数
        
        Args:
            question: 用户问题
            
        Returns:
            答案字符串
        """
        print(f"问题：{question}")
        print("-" * 60)
        
        # 1. 提取问题中的故障类型
        fault_name = self._extract_fault_name(question)
        
        if not fault_name:
            return "抱歉，我没有理解您的问题。请 specify 具体的故障类型，例如：刀具磨损、主轴过热、导轨爬行等。"
        
        print(f"识别到故障类型：{fault_name}")
        
        # 2. 根据问题类型，查询不同信息
        if "原因" in question or "为什么" in question:
            return self._answer_cause(fault_name)
        elif "症状" in question or "现象" in question or "表现" in question:
            return self._answer_symptom(fault_name)
        elif "解决" in question or "处理" in question or "修复" in question or "怎么办" in question:
            return self._answer_solution(fault_name)
        elif "检查" in question or "检测" in question or "诊断" in question:
            return self._answer_check(fault_name)
        elif "预防" in question or "避免" in question or "防止" in question:
            return self._answer_prevention(fault_name)
        elif "参数" in question or "调整" in question or "设置" in question:
            return self._answer_parameter(fault_name)
        elif "材料" in question or "加工" in question:
            return self._answer_material(fault_name)
        elif "部件" in question or "组件" in question or "零件" in question:
            return self._answer_component(fault_name)
        elif "类别" in question or "分类" in question or "类型" in question:
            return self._answer_category(fault_name)
        elif "修复方式" in question or "治疗方法" in question:
            return self._answer_cure_way(fault_name)
        elif "容易" in question or "发生" in question or "情况" in question:
            return self._answer_easy_get(fault_name)
        elif "时间" in question or "多久" in question:
            return self._answer_cure_lasttime(fault_name)
        elif "概率" in question or "能修好" in question or "成功率" in question:
            return self._answer_cured_prob(fault_name)
        else:
            # 默认：返回故障的详细信息
            return self._answer_full_info(fault_name)
    
    def _extract_fault_name(self, question):
        """
        从问题中提取故障类型名称
        
        改进：直接从数据库查询所有故障类型，然后匹配
        """
        # 查询数据库中的所有故障类型
        result = self.graph.run("MATCH (f:FaultType) RETURN f.name AS name")
        fault_names = [record["name"] for record in result.data()]
        
        # 在问题中搜索故障类型
        for name in fault_names:
            if name in question:
                return name
        
        # 如果没有精确匹配，返回 None
        return None
    
    def _answer_cause(self, fault_name):
        """回答故障原因"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_CAUSE]->(c:Cause)
        RETURN c.text AS cause
        """
        result = self.graph.run(query, name=fault_name)
        causes = [record["cause"] for record in result.data()]
        
        if not causes:
            return f"抱歉，知识库中没有关于【{fault_name}】的原因信息。"
        
        answer = f"【{fault_name}】的常见原因包括：\n"
        for i, cause in enumerate(causes, 1):
            answer += f"{i}. {cause}\n"
        return answer
    
    def _answer_symptom(self, fault_name):
        """回答故障症状"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.text AS symptom
        """
        result = self.graph.run(query, name=fault_name)
        symptoms = [record["symptom"] for record in result.data()]
        
        if not symptoms:
            return f"抱歉，知识库中没有关于【{fault_name}】的症状信息。"
        
        answer = f"【{fault_name}】的常见症状包括：\n"
        for i, symptom in enumerate(symptoms, 1):
            answer += f"{i}. {symptom}\n"
        return answer
    
    def _answer_solution(self, fault_name):
        """回答故障解决方案"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_SOLUTION]->(s:Solution)
        RETURN s.text AS solution
        """
        result = self.graph.run(query, name=fault_name)
        solutions = [record["solution"] for record in result.data()]
        
        if not solutions:
            return f"抱歉，知识库中没有关于【{fault_name}】的解决方案。"
        
        answer = f"【{fault_name}】的解决方案包括：\n"
        for i, solution in enumerate(solutions, 1):
            answer += f"{i}. {solution}\n"
        return answer
    
    def _answer_check(self, fault_name):
        """回答故障检查方法"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:NEEDS_CHECK]->(c:Check)
        RETURN c.text AS check
        """
        result = self.graph.run(query, name=fault_name)
        checks = [record["check"] for record in result.data()]
        
        if not checks:
            return f"抱歉，知识库中没有关于【{fault_name}】的检查方法。"
        
        answer = f"【{fault_name}】的检查方法包括：\n"
        for i, check in enumerate(checks, 1):
            answer += f"{i}. {check}\n"
        return answer
    
    def _answer_prevention(self, fault_name):
        """回答故障预防措施"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_PREVENT]->(p:Prevent)
        RETURN p.text AS prevention
        """
        result = self.graph.run(query, name=fault_name)
        preventions = [record["prevention"] for record in result.data()]
        
        if not preventions:
            return f"抱歉，知识库中没有关于【{fault_name}】的预防措施。"
        
        answer = f"【{fault_name}】的预防措施包括：\n"
        for i, prevention in enumerate(preventions, 1):
            answer += f"{i}. {prevention}\n"
        return answer
    
    def _answer_parameter(self, fault_name):
        """回答故障相关参数"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_PARAMETER]->(p:Parameter)
        RETURN p.param_name AS param_name, p.value AS value, p.adjustment AS adjustment
        """
        result = self.graph.run(query, name=fault_name)
        params = result.data()
        
        if not params:
            return f"抱歉，知识库中没有关于【{fault_name}】的参数信息。"
        
        answer = f"【{fault_name}】的相关参数设置：\n"
        for i, param in enumerate(params, 1):
            answer += f"{i}. {param['param_name']}: {param['value']}"
            if param['adjustment']:
                answer += f" (调整建议: {param['adjustment']})"
            answer += "\n"
        return answer
    
    def _answer_material(self, fault_name):
        """回答故障适用的材料"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:APPLIES_TO_MATERIAL]->(m:Material)
        RETURN m.text AS material
        """
        result = self.graph.run(query, name=fault_name)
        materials = [record["material"] for record in result.data()]
        
        if not materials:
            return f"抱歉，知识库中没有关于【{fault_name}】的适用材料信息。"
        
        answer = f"【{fault_name}】常见于以下材料：\n"
        for i, material in enumerate(materials, 1):
            answer += f"{i}. {material}\n"
        return answer
    
    def _answer_component(self, fault_name):
        """回答故障涉及的部件"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:INVOLVES_COMPONENT]->(c:Component)
        RETURN c.text AS component
        """
        result = self.graph.run(query, name=fault_name)
        components = [record["component"] for record in result.data()]
        
        if not components:
            return f"抱歉，知识库中没有关于【{fault_name}】的涉及部件信息。"
        
        answer = f"【{fault_name}】涉及的部件包括：\n"
        for i, component in enumerate(components, 1):
            answer += f"{i}. {component}\n"
        return answer
    
    def _answer_category(self, fault_name):
        """回答故障类别"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:BELONGS_TO_CATEGORY]->(c:Category)
        RETURN c.text AS category
        """
        result = self.graph.run(query, name=fault_name)
        categories = [record["category"] for record in result.data()]
        
        if not categories:
            return f"抱歉，知识库中没有关于【{fault_name}】的类别信息。"
        
        answer = f"【{fault_name}】属于以下类别：\n"
        for i, category in enumerate(categories, 1):
            answer += f"{i}. {category}\n"
        return answer
    
    def _answer_cure_way(self, fault_name):
        """回答故障的治疗方式"""
        query = """
        MATCH (f:FaultType {name: $name})-[r:HAS_CURE_WAY]->(c:CureWay)
        RETURN c.text AS cure_way
        """
        result = self.graph.run(query, name=fault_name)
        cure_ways = [record["cure_way"] for record in result.data()]
        
        if not cure_ways:
            return f"抱歉，知识库中没有关于【{fault_name}】的治疗方式信息。"
        
        answer = f"【{fault_name}】的治疗方式包括：\n"
        for i, cure_way in enumerate(cure_ways, 1):
            answer += f"{i}. {cure_way}\n"
        return answer
    
    def _answer_easy_get(self, fault_name):
        """回答故障的易发情况"""
        query = """
        MATCH (f:FaultType {name: $name})
        RETURN f.easy_get AS easy_get
        """
        result = self.graph.run(query, name=fault_name)
        data = result.data()
        
        if not data or not data[0]["easy_get"]:
            return f"抱歉，知识库中没有关于【{fault_name}】的易发情况信息。"
        
        easy_get = data[0]["easy_get"]
        return f"【{fault_name}】在以下情况下更容易发生：\n{easy_get}"
    
    def _answer_cure_lasttime(self, fault_name):
        """回答故障的修复时间"""
        query = """
        MATCH (f:FaultType {name: $name})
        RETURN f.cure_lasttime AS cure_lasttime
        """
        result = self.graph.run(query, name=fault_name)
        data = result.data()
        
        if not data or not data[0]["cure_lasttime"]:
            return f"抱歉，知识库中没有关于【{fault_name}】的修复时间信息。"
        
        cure_lasttime = data[0]["cure_lasttime"]
        return f"【{fault_name}】的修复时间：\n{cure_lasttime}"
    
    def _answer_cured_prob(self, fault_name):
        """回答故障的修复概率"""
        query = """
        MATCH (f:FaultType {name: $name})
        RETURN f.cured_prob AS cured_prob
        """
        result = self.graph.run(query, name=fault_name)
        data = result.data()
        
        if not data or not data[0]["cured_prob"]:
            return f"抱歉，知识库中没有关于【{fault_name}】的修复概率信息。"
        
        cured_prob = data[0]["cured_prob"]
        return f"【{fault_name}】的修复成功率：\n{cured_prob}"
    
    def _answer_full_info(self, fault_name):
        """回答故障的所有详细信息"""
        query = """
        MATCH (f:FaultType {name: $name})
        RETURN f
        """
        result = self.graph.run(query, name=fault_name)
        data = result.data()
        
        if not data:
            return f"抱歉，知识库中没有关于【{fault_name}】的信息。"
        
        f = data[0]["f"]
        
        answer = f"【{fault_name}】的详细信息：\n"
        answer += f"描述：{f['desc']}\n\n"
        
        # 查询所有相关信息
        answer += self._answer_cause(fault_name) + "\n"
        answer += self._answer_symptom(fault_name) + "\n"
        answer += self._answer_solution(fault_name) + "\n"
        answer += self._answer_check(fault_name) + "\n"
        answer += self._answer_prevention(fault_name) + "\n"
        answer += self._answer_parameter(fault_name) + "\n"
        answer += self._answer_component(fault_name) + "\n"
        answer += self._answer_material(fault_name) + "\n"
        answer += self._answer_category(fault_name) + "\n"
        answer += self._answer_cure_way(fault_name) + "\n"
        answer += self._answer_easy_get(fault_name) + "\n"
        answer += self._answer_cure_lasttime(fault_name) + "\n"
        answer += self._answer_cured_prob(fault_name) + "\n"
        
        return answer


def main():
    """交互式测试"""
    print("=" * 70)
    print("  机床故障诊断问答系统 - 简化版")
    print("  Machine Fault Diagnosis Q&A System - Simplified Version")
    print("=" * 70)
    print()
    
    # 初始化问答机器人
    try:
        bot = SimpleChatBot(neo4j_password="200980216")
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        print("请检查：")
        print("1. Neo4j 是否已启动")
        print("2. 密码是否正确")
        return
    
    print("=" * 70)
    print("交互式问答模式")
    print("=" * 70)
    print()
    print("示例问题：")
    print("  - 刀具磨损是什么原因？")
    print("  - 主轴过热有什么症状？")
    print("  - 如何解决崩刃问题？")
    print("  - 导轨爬行的检查方法是什么？")
    print("  - 刀具磨损的预防措有哪些？")
    print("  - 刀具磨损的参数如何调整？")
    print("  - 刀具磨损常见于什么材料？")
    print("  - 刀具磨损涉及哪些部件？")
    print("  - 刀具磨损属于什么类别？")
    print("  - 刀具磨损的修复方式有哪些？")
    print("  - 刀具磨损在什么情况下容易发生？")
    print("  - 刀具磨损修复需要多长时间？")
    print("  - 刀具磨损能修好吗？")
    print("  - 详细介绍刀具磨损")
    print()
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 70)
    print()
    
    while True:
        try:
            question = input("问题：").strip()
            
            if not question:
                continue
                
            if question.lower() in ["quit", "exit", "q", "退出"]:
                print("感谢使用，再见！")
                break
            
            answer = bot.ask(question)
            print()
            print("答案：")
            print(answer)
            print()
            print("-" * 70)
            print()
            
        except KeyboardInterrupt:
            print("\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 处理出错：{e}")
            continue


if __name__ == "__main__":
    main()
