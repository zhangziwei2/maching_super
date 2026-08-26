"""
手工/专家三元组图谱（Entity:Upload）—— JSONL 三元组 → Neo4j + 节点向量化 + 子图检索。
参考 Yuxi upload_graph_service 精简实现：实体向量直接存 Neo4j 节点（entityEmbeddings 向量索引），
检索 = 向量命中 + 模糊匹配 → 1 跳子图扩展。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# embedding_service 位于 backend/embedding.py；兼容「backend 作为顶层包」与
# 「backend 作为顶级模块（python -m app 于 backend/ 下）」两种启动方式
try:
    from ..embedding import embedding_service
except ImportError:
    from embedding import embedding_service

logger = logging.getLogger("graphkb")

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_VECTOR_INDEX = "entityEmbeddings"
_DIM = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

# 子图扩展时排除的工具/操作类关系（与 kg/graph_schema.GRAPH_NOISE_REL_TYPES 保持一致）：
# 这类关系不承载诊断因果语义，多种子扩展时会混入"更换主轴轴承 -[需要工具]-> 轴承拉拔器"
# 之类与故障本身无关的实体关系。
_NOISE_REL_TYPES = frozenset({"使用工具", "需要工具"})


class UploadGraph:
    """手工三元组图谱：写入 Neo4j（Entity:Upload）并提供实体检索。"""

    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "lightrag2026")
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ---------- 写入 ----------

    def add_triples(self, triples: list[dict]) -> dict:
        if not triples:
            return {"imported": 0, "vectorized": 0}
        names: set[str] = set()
        with self.driver.session() as s:
            for t in triples:
                h, r, tt = (t.get("h") or {}), (t.get("r") or {}), (t.get("t") or {})
                hn = str((h.get("name") or "")).strip()
                tn = str((tt.get("name") or "")).strip()
                rt = str((r.get("type") or "RELATED_TO")).strip()
                if not hn or not tn:
                    continue
                names.add(hn)
                names.add(tn)
                s.run(
                    """
                    MERGE (h:Entity:Upload {name: $hn}) SET h.type = $ht
                    MERGE (t:Entity:Upload {name: $tn}) SET t.type = $tt
                    MERGE (h)-[r:RELATION {type: $rt}]->(t)
                    """,
                    hn=hn,
                    ht=str((h.get("type") or "Entity")).strip(),
                    tn=tn,
                    tt=str((tt.get("type") or "Entity")).strip(),
                    rt=rt,
                )
            self._ensure_index(s)
            vec = self._fill(s, sorted(names))
        return {"imported": len(triples), "vectorized": vec}

    def _ensure_index(self, session) -> None:
        try:
            exists = any(r["name"] == _VECTOR_INDEX for r in session.run("SHOW INDEXES"))
            if not exists:
                session.run(
                    f"CREATE VECTOR INDEX `{_VECTOR_INDEX}` FOR (n:Entity) ON (n.embedding) "
                    f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {_DIM}, `vector.similarity_function`: 'cosine' }} }}"
                )
        except Exception as e:
            logger.warning("[UploadGraph] 向量索引创建失败（向量检索将降级为模糊匹配）: %s", e)

    def _fill(self, session, names: list[str]) -> int:
        if not names:
            return 0
        rows = session.run(
            "MATCH (n:Entity:Upload) WHERE n.name IN $names AND n.embedding IS NULL RETURN n.name AS name",
            names=names,
        ).data()
        missing = [r["name"] for r in rows]
        if not missing:
            return 0
        try:
            vecs = embedding_service.get_embeddings(missing)
        except Exception as e:
            logger.warning("[UploadGraph] 节点向量化失败（检索降级为模糊匹配）: %s", e)
            return 0
        n = 0
        for name, vec in zip(missing, vecs):
            session.run(
                "MATCH (e:Entity:Upload {name: $name}) CALL db.create.setNodeVectorProperty(e, 'embedding', $vec)",
                name=name,
                vec=vec,
            )
            n += 1
        return n

    # ---------- 检索 ----------

    def query_node(self, keyword: str, hops: int = 2, top_k: int = 5) -> dict:
        kw = (keyword or "").strip()
        if not kw:
            return {"nodes": [], "edges": [], "triples": []}
        scores: dict[str, float] = {}
        # 1) 向量检索（索引存在时）
        try:
            vec = embedding_service.get_embeddings([kw])[0]
            with self.driver.session() as s:
                rows = s.run(
                    f"CALL db.index.vector.queryNodes('{_VECTOR_INDEX}', {top_k}, $vec) YIELD node AS n, score "
                    f"WHERE 'Upload' IN labels(n) RETURN n.name AS name, score",
                    vec=vec,
                ).data()
            for r in rows:
                scores[r["name"]] = max(scores.get(r["name"], 0.0), float(r["score"]))
        except Exception as e:
            logger.debug("[UploadGraph] 向量检索失败，降级模糊匹配: %s", e)
        # 2) 模糊匹配（兜底）
        try:
            with self.driver.session() as s:
                rows = s.run(
                    "MATCH (n:Entity:Upload) WHERE toLower(n.name) CONTAINS toLower($kw) RETURN DISTINCT n.name AS name",
                    kw=kw,
                ).data()
            for r in rows:
                scores[r["name"]] = max(scores.get(r["name"], 0.0), 0.3)
        except Exception as e:
            logger.debug("[UploadGraph] 模糊匹配失败: %s", e)
        # 去重种子名（向量命中与模糊命中可能指向同一实体）
        seen_q: set[str] = set()
        qualified: list[str] = []
        for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if n not in seen_q:
                seen_q.add(n)
                qualified.append(n)
        return self._expand(qualified)

    def _expand(self, names: list[str]) -> dict:
        result = {"nodes": [], "edges": [], "triples": []}
        for name in names:
            try:
                with self.driver.session() as s:
                    # 种子节点本身（即使无边也要返回，保证图谱连通）
                    seed = s.run(
                        "MATCH (n:Upload {name: $name}) RETURN n.name AS name, n.type AS type",
                        name=name,
                    ).data()
                    for r in seed:
                        result["nodes"].append({"name": r["name"], "type": r.get("type"), "source": "upload"})
                    # 关系与邻居：用 startNode/endNode 取真实方向，r.type 为语义关系类型
                    rows = s.run(
                        """
                        MATCH (n:Upload {name: $name})-[r]-(m)
                        RETURN startNode(r).name AS rs, startNode(r).type AS rst,
                               r.type AS rt, endNode(r).name AS re, endNode(r).type AS ret
                        LIMIT 50
                        """,
                        name=name,
                    ).data()
                for r in rows:
                    rs, re, rt = r["rs"], r["re"], r["rt"]
                    if rt in _NOISE_REL_TYPES:
                        continue
                    result["nodes"].append({"name": rs, "type": r.get("rst"), "source": "upload"})
                    result["nodes"].append({"name": re, "type": r.get("ret"), "source": "upload"})
                    result["edges"].append({"source": rs, "target": re, "type": rt})
                    result["triples"].append([rs, rt, re])
            except Exception as e:
                logger.debug("[UploadGraph] 子图扩展失败: %s", e)
        # 去重
        seen_n: set[str] = set()
        nodes = []
        for n in result["nodes"]:
            if n["name"] not in seen_n:
                seen_n.add(n["name"])
                nodes.append(n)
        seen_e: set[tuple] = set()
        edges = []
        for e in result["edges"]:
            key = (e["source"], e["type"], e["target"])
            if key not in seen_e:
                seen_e.add(key)
                edges.append(e)
        seen_t: set[tuple] = set()
        triples = []
        for t in result["triples"]:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                key = (t[0], t[1], t[2])
                if key not in seen_t:
                    seen_t.add(key)
                    triples.append(t)
        return {"nodes": nodes, "edges": edges, "triples": triples}
