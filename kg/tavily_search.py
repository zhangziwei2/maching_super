# -*- coding: utf-8 -*-
"""
机床故障诊断 - Tavily联网搜索模块
当知识图谱中无法找到答案时，通过Tavily搜索引擎从互联网获取实时信息。
"""

import os
import requests
from dotenv import load_dotenv

#  加载 .env 文件（必须在同目录）
load_dotenv()


class TavilySearch:
    def __init__(self):
        #  直接从环境变量读取
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        self.max_results = 5
        self.search_depth = "advanced"
        self.include_answer = True
        self.base_url = "https://api.tavily.com/search"

    def search(self, query, context="机床故障诊断"):
        """
        使用Tavily API进行联网搜索
        """
        if not self.api_key:
            print(" 未找到 TAVILY_API_KEY，请检查 .env 文件")
            return None

        # 增强查询词
        enhanced_query = f"{query} {context}"

        payload = {
            "api_key": self.api_key,
            "query": enhanced_query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
            "include_raw_content": False,
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )

            #  打印状态码方便调试
            if response.status_code == 401:
                print(" API Key 无效（401）")
                return None

            response.raise_for_status()
            data = response.json()

            # 提取答案
            answer = data.get("answer", "")
            results = data.get("results", [])

            if answer:
                formatted = answer
            elif results:
                formatted = self._format_results(results)
            else:
                return "未找到相关结果"

            # 添加来源引用
            sources = []
            for r in results[:3]:
                title = r.get("title", "")
                url = r.get("url", "")
                if title and url:
                    sources.append(f"- {title}: {url}")

            if sources:
                formatted += "\n\n 参考来源：\n" + "\n".join(sources)

            return formatted

        except requests.exceptions.RequestException as e:
            print(f"[Tavily] 搜索请求失败: {e}")
            return None
        except Exception as e:
            print(f"[Tavily] 处理失败: {e}")
            return None

    def _format_results(self, results):
        """格式化搜索结果为多段文本"""
        parts = []
        for i, r in enumerate(results[:3], 1):
            content = r.get("content", "")
            if content:
                parts.append(f"{i}. {content[:200]}...")
        return "\n".join(parts)

    def is_configured(self):
        """检查是否已配置API Key"""
        return bool(self.api_key)


if __name__ == '__main__':
    ts = TavilySearch()
    if ts.is_configured():
        result = ts.search("主轴轴承发热故障原因")
        print(" 搜索成功：\n")
        print(result)
    else:
        print(" 请先在 .env 文件中配置 TAVILY_API_KEY")
