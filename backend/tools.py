# -*- coding: utf-8 -*-
"""
工具定义 - 集成知识图谱、RAG 检索、联网搜索、天气查询
"""
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv
from typing import Optional

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, PROJECT_ROOT)

# 添加 kg 目录到路径，以便 chatbot_graph 等模块的绝对导入正常工作
KG_DIR = os.path.join(PROJECT_ROOT, "kg")
if os.path.exists(KG_DIR) and KG_DIR not in sys.path:
    sys.path.insert(0, KG_DIR)

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None  # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass


# ========== 工具定义 ==========

try:
    from langchain_core.tools import tool
except ImportError:
    try:
        from langchain_core.tools import tool
    except ImportError:
        def tool(name_or_func=None, *args, **kwargs):
            def decorator(func):
                func._is_tool = True
                func._tool_name = name_or_func if isinstance(name_or_func, str) else func.__name__
                return func
            if callable(name_or_func):
                return decorator(name_or_func)
            return decorator


@tool("query_knowledge_graph")
def query_knowledge_graph(query: str) -> str:
    """
    查询机床故障诊断知识图谱（统一图谱：领域规则图谱 + 手工三元组 + LightRAG 自动图谱，
    按 GRAPH_FUSION / GRAPH_SOURCES 程度开关融合）。

    当用户询问机床故障、症状、原因、解决方法、预防措施、参数调整等问题时，
    使用此工具查询知识图谱。

    适用问题类型：
    - 故障诊断：主轴过热是什么原因？
    - 症状判断：铣削时出现振纹是什么故障？
    - 解决方法：刀具崩刃怎么处理？
    - 预防措施：如何避免刀具磨损？
    - 参数调整：表面粗糙度差该怎么调参数？
    - 检测手段：怎么检测主轴是否过热？
    - 部件故障：导轨一般会出现什么故障？

    返回的是「实体 -[关系]-> 实体」三元组事实 + 故障属性信息，供你推理组织答案。

    Args:
        query: 用户问题（中文）

    Returns:
        知识图谱三元组文本，未找到信息时返回空字符串
    """
    try:
        from .graphkb import get_graph_kb
        text = get_graph_kb().query_text(query)
        return text or ""
    except Exception:  # noqa: BLE001
        # 兜底：退回 legacy NetworkX 图谱，保证工具不硬失败
        try:
            from kg.graph_service import graph_service
            graph_service.ensure_ready()
            text = graph_service.query_text(query)
            return text or ""
        except Exception:
            return ""


@tool("web_search")
def web_search(query: str) -> str:
    """
    联网搜索获取实时信息。
    
    当知识图谱和 RAG 都无法回答问题时，使用此工具联网搜索。
    适用于：实时信息、最新技术、网络资料等。
    
    Args:
        query: 搜索关键词（中文）
        
    Returns:
        搜索结果摘要
    """
    try:
        from kg.tavily_search import TavilySearch
        searcher = TavilySearch()
        if not searcher.is_configured():
            return "联网搜索未配置（缺少 TAVILY_API_KEY）"
        result = searcher.search(query, context="机床故障诊断")
        if not result:
            return "联网搜索未找到相关结果。"
        return result
    except Exception as e:
        return f"联网搜索失败: {str(e)}"


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """
    从文档知识库检索相关段落（Milvus 混合检索 + 父子分块 auto-merging + 查询改写）。
    当用户询问操作手册、设备使用步骤、技术文档内容时使用此工具。
    """
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    try:
        from .rag_pipeline import run_rag_graph
        rag_result = run_rag_graph(query)
    except Exception as e:
        return f"知识库检索失败: {e}"

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("diagnose_chatter")
def diagnose_chatter(csv_path: str, mode: str = "chatter_comprehensive") -> str:
    """
    对机床传感器数据进行颤振诊断，支持多种判断模式。

    输入 CSV/XLSX 文件格式（5列）：时间(秒), 主轴振动, X轴振动, Y轴振动, 三向力合力
    采样点数要求：至少 512 个点。
    输出：包含诊断状态（稳定/轻微颤振/严重颤振）、置信度、分析依据和工艺建议的完整诊断报告。

    诊断模式 mode 参数：
      chatter_comprehensive - 综合诊断（默认）：同时运行全部模式投票决策
      chatter_amplitude     - 幅值阈值模式：基于 RMS/峰值幅值阈值判断
      chatter_frequency     - 频谱特征模式：基于功率谱峰值/频率方差判断
      chatter_timefreq      - 时频分析模式：基于 STFT 时频谱能量/熵判断
      chatter_fusion        - 多源融合模式：SAE + stacking 综合诊断
      chatter_trend         - 趋势监测模式：滑动窗口特征趋势判断
      chatter_monitor       - 实时监控模式：基于设备专属基线，z-score 偏离超阈值报警

    使用场景：
    - 用户提供传感器数据文件需要诊断是否发生颤振
    - 用户询问加工过程是否稳定
    - 需要分析切削颤振状态

    Args:
        csv_path: CSV/XLSX 文件的服务器路径
        mode: 诊断模式，默认 chatter_comprehensive

    Returns:
        诊断报告文本
    """
    try:
        from .chatter.chatter_diagnosis_skill import diagnose_csv
        if not os.path.exists(csv_path):
            return f"文件不存在: {csv_path}"
        report = diagnose_csv(csv_path, mode=mode)
        return report
    except FileNotFoundError as e:
        return (
            f"模型文件未找到: {e}\n"
            f"请先运行训练脚本生成模型:\n"
            f"  python -m backend.chatter.train_all"
        )
    except ImportError as e:
        return (
            f"颤振诊断模块导入失败: {e}\n"
            f"请确认已安装依赖:\n"
            f"  pip install torch numpy pandas scipy scikit-learn joblib openpyxl"
        )
    except Exception as e:
        return f"颤振诊断执行失败: {e}"

