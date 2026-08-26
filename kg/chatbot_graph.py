# -*- coding: utf-8 -*-
"""
机床故障诊断GraphRAG知识图谱问答系统 - 主入口
全字段建模版本：支持10种节点类型、10种关系类型、综合查询（NetworkX 全内嵌）

v2.0：移除 Neo4j/py2neo 依赖，图谱引擎切换为 NetworkX（graph_store.py）。
整合意图识别、查询生成、答案搜索、Tavily联网搜索、GraphRAG增强
预留传感器实时数据接口

支持20种意图类型：
1. 故障原因查询      e.g. "主轴过热是什么原因？"
2. 现象诊断          e.g. "铣削时出现振纹是什么故障？"
3. 解决方法推荐      e.g. "刀具崩刃怎么处理？"
4. 部件关联故障      e.g. "导轨一般会出现什么故障？"
5. 预防措施          e.g. "如何避免刀具磨损？"
6. 参数优化          e.g. "表面粗糙度差该怎么调参数？"
7. 检测手段          e.g. "怎么检测主轴是否过热？"
8. 故障类别查询      e.g. "刀具磨损属于什么类别？"
9. 修复方式查询      e.g. "主轴过热该怎么修复？"
10. 易发情况查询     e.g. "刀具磨损在什么情况下容易发生？"
11. 修复时间查询     e.g. "主轴过热修复需要多长时间？"
12. 修复概率查询     e.g. "刀具磨损能修好吗？"
13. 综合查询（故障全信息） e.g. "详细介绍刀具磨损的所有信息"
14. 多跳查询：症状故障原因解决方案
15. 多跳查询：部件故障检测方法
16. 多跳查询：材料故障预防措施
"""

from question_classifier import QuestionClassifier
from question_parser import QuestionParser
from answer_search import AnswerSearcher
from config import CHATBOT_CONFIG


class MachineFaultChatBot:
    """机床故障诊断聊天机器人 - 全字段建模版本（NetworkX 版）"""

    def __init__(self):
        print("=" * 60)
        print("  机床故障诊断 GraphRAG 知识图谱问答系统 (NetworkX 全内嵌版)")
        print("  Machine Fault Diagnosis KG Q&A System (Embedded NetworkX)")
        print("=" * 60)

        # 初始化各个组件
        try:
            self.classifier = QuestionClassifier()
            print(" 意图识别模块加载成功")
        except Exception as e:
            print(f" 意图识别模块加载失败: {e}")
            raise

        try:
            self.parser = QuestionParser()
            print(" 查询解析模块加载成功")
        except Exception as e:
            print(f" 查询解析模块加载失败: {e}")
            raise

        try:
            self.searcher = AnswerSearcher()
            print(" 答案搜索模块加载成功")
        except Exception as e:
            print(f" 答案搜索模块加载失败: {e}")
            raise

        # 图谱统一检索服务（供 Agent 工具 / RAG 融合使用）
        try:
            from graph_service import graph_service
            graph_service.ensure_ready()
            self.graph_service = graph_service
            stats = graph_service.stats()
            print(f" 知识图谱加载成功: {stats.get('total_nodes', 0)} 节点, {stats.get('total_edges', 0)} 边")
        except Exception as e:
            print(f" 知识图谱加载失败: {e}")
            self.graph_service = None

        self.default_answer = CHATBOT_CONFIG["default_answer"]
        print("\n系统初始化完成！\n")

    def chat_main(self, sent):
        """
        主问答流程：
        1. 意图识别
        2. 查询生成
        3. 知识图谱查询（NetworkX）
        4. 返回格式化答案
        """
        # 步骤1: 意图识别
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return self.default_answer

        # 步骤2: 查询生成
        res_sql = self.parser.parser_main(res_classify)
        if not res_sql:
            return self.default_answer

        # 步骤3: 知识图谱查询
        final_answers = self.searcher.search_main(res_sql)

        if not final_answers or len(final_answers) == 0:
            return self.default_answer

        return '\n\n'.join(final_answers)

    def query_graph(self, query: str, hops: int = 2):
        """
        结构化图谱查询（供 Agent 工具 / API 调用）。
        返回 {"nodes": [...], "edges": [...], "triples": [...]}
        """
        if self.graph_service is None:
            return {"nodes": [], "edges": [], "triples": [], "seed_entities": [], "seed_namespace": "", "query": query, "fallback": True}
        return self.graph_service.query(query, hops=hops)

    def get_fault_full_info(self, fault_name):
        """
        综合查询：一次性获取故障的所有信息
        这是一个单独的API接口，可以直接调用
        """
        sent = f"{fault_name} 详细信息"
        return self.chat_main(sent)

    def run_interactive(self):
        """交互式运行"""
        print("\n已进入交互模式，输入问题即可（输入 'quit' 退出）：\n")
        print("示例问题：")
        print("  - 主轴过热是什么原因？")
        print("  - 铣削时出现振纹是什么故障？")
        print("  - 刀具崩刃怎么处理？")
        print("  - 导轨一般会出现什么故障？")
        print("  - 如何避免刀具磨损？")
        print("  - 表面粗糙度差该怎么调参数？")
        print("  - 怎么检测主轴是否过热？")
        print("  - 刀具磨损属于什么类别？")
        print("  - 主轴过热该怎么修复？")
        print("  - 刀具磨损在什么情况下容易发生？")
        print("  - 主轴过热修复需要多长时间？")
        print("  - 刀具磨损能修好吗？")
        print("  - 详细介绍刀具磨损的所有信息")
        print()

        while True:
            try:
                question = input('咨询: ').strip()
                if not question:
                    continue
                if question.lower() in ['quit', 'exit', 'q', '退出']:
                    print("感谢使用，再见！")
                    break

                answer = self.chat_main(question)
                print('客服机器人:', answer)
                print()

            except KeyboardInterrupt:
                print("\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"处理出错: {e}")
                continue


if __name__ == '__main__':
    try:
        handler = MachineFaultChatBot()
        handler.run_interactive()
    except Exception as e:
        print(f"\n程序启动失败: {e}")
        input("\n按回车键退出...")
