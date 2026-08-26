# -*- coding: utf-8 -*-
"""
机床故障诊断GraphRAG知识图谱问答系统 - 配置文件
Machine Fault Diagnosis GraphRAG KG Q&A System - Configuration

v2.0：移除 Neo4j 依赖，改为全内嵌 NetworkX 图引擎（见 graph_store.py）
"""

import os

# ========== 知识图谱配置（全内嵌，无外部图数据库） ==========
GRAPH_CONFIG = {
    "enabled": True,
    "data_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "machine_fault.json"),
    "triples_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "triples"),
    "graph_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "graph_cache.pkl"),
    "hops": 2,          # 子图扩展跳数（对标 Yuxi 的 1~2 跳）
    "top_k": 10,        # 种子实体召回数
    "fallback": True,   # 实体未命中时是否启用 CONTAINS 子串兜底
    # 构建图谱时过滤的故障名黑名单（剔除混入的脏数据，原始数据文件不动）
    "exclude_fault_names": [
        "肺泡蛋白质沉积症",
        "百日咳",
        "恒益药业琥乙红霉素颗粒",
    ],
}

# ========== Tavily 联网搜索配置 ==========
TAVILY_CONFIG = {
    "api_key": os.getenv("TAVILY_API_KEY", "ly-dev-482SnJ-wkW31Zq4gHI8PNM8d9ypOzKA3FXZPcFwjJaNxUuCUa"),  # 请替换为实际Tavily API Key
    "max_results": 5,
    "search_depth": "advanced",
    "include_answer": True,
    "include_raw_content": False,
}

# ========== 传感器实时数据接口配置 ==========
SENSOR_CONFIG = {
    "enabled": False,  # 默认关闭，接入传感器后设为True
    "websocket_url": "ws://localhost:8765",  # WebSocket传感器数据流地址
    "http_endpoint": "http://localhost:8080/api/v1/sensor/data",  # HTTP轮询端点
    "poll_interval": 1.0,  # HTTP轮询间隔(秒)
    "vibration_channels": ["X", "Y", "Z"],  # 振动传感器通道
    "force_channels": ["Fx", "Fy", "Fz"],   # 力传感器通道
    "sound_level_channel": "SPL",            # 声级计通道
    "alarm_thresholds": {
        "vibration_rms": 5.0,     # 振动有效值报警阈值 (m/s^2)
        "spike_factor": 3.0,      # 冲击因子阈值
        "sound_level": 85.0,      # 声级报警阈值 (dB)
        "temperature": 70.0,      # 温度报警阈值 (C)
    },
    "feature_extraction": {
        "window_size": 1024,
        "overlap": 512,
        "sample_rate": 10000,  # Hz
    }
}

# ========== GraphRAG 配置 ==========
GRAPHRAG_CONFIG = {
    "enabled": True,
    "embedding_model": "BAAI/bge-m3",  # 本地模型，与 backend/embedding.py 对齐
    "vector_dim": 1024,
    "similarity_threshold": 0.75,
    "top_k": 3,
    "chunk_size": 200,
    "chunk_overlap": 50,
}

# ========== 项目路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "machine_fault.json")
DICT_DIR = os.path.join(BASE_DIR, "dict")

# ========== 问答系统配置 ==========
CHATBOT_CONFIG = {
    "num_limit": 20,           # 返回结果数量限制
    "default_answer": "没能理解您的问题，我的知识库还在完善中。您可以尝试用更标准的方式提问，例如：\n"
                       "- '主轴过热是什么原因？'\n"
                       "- '铣削时出现振纹是什么故障？'\n"
                       "- '刀具崩刃怎么处理？'\n"
                       "- '导轨一般会出现什么故障？'\n"
                       "- '如何避免刀具磨损？'\n"
                       "- '表面粗糙度差该怎么调参数？'\n"
                       "- '怎么检测主轴是否过热？'",
    "enable_tavily_fallback": True,  # 知识图谱无结果时是否启用Tavily搜索
}
