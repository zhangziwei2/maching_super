"""
统一图谱检索包（Phase 2）。

对外暴露：
  - GraphKB / get_graph_kb()  统一图谱服务，按 GRAPH_FUSION / GRAPH_SOURCES 融合三来源

三来源：
  - domain         领域规则图谱（machine_fault.json）→ 沿用成熟 NetworkX kg.graph_service
  - manual_triples 手工/专家三元组 JSONL            → Neo4j Entity:Upload（本包 upload_graph）
  - lightrag       文档自动抽取图谱                 → LightRAG（Neo4JStorage，若可用）
"""
from .service import GraphKB, get_graph_kb

__all__ = ["GraphKB", "get_graph_kb"]
