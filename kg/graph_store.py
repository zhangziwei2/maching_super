# -*- coding: utf-8 -*-
"""
机床故障诊断知识图谱 - NetworkX 图引擎（替代 Neo4j 的全内嵌存储）

职责：
1. 从 machine_fault.json 构建规则图谱（英文标签 schema，对齐历史 question_parser 模板）
2. 加载 data/triples/*.json / *.jsonl 人工导入三元组，构建 Upload 子图
3. pickle 持久化 / 全量重建（单一事实源：JSON + JSONL，pickle 仅作启动加速缓存）
4. 提供图遍历查询原语：节点定位、多跳子图扩展、邻居聚合

设计要点：
- 节点 id 使用三元组 (namespace, entity_type, name)，避免跨命名空间/类型同名冲突；
- 多边图（MultiDiGraph），同一对节点间可存在多条不同 rel_type 的边；
- 线程安全：所有写操作加锁，供 FastAPI 并发调用。
"""

import json
import os
import pickle
import threading
from typing import Optional

import networkx as nx

from graph_schema import (
    NAMESPACE_RULE,
    NAMESPACE_UPLOAD,
    RULE_NODE_TYPES,
    FAULT_TYPE_PROPS,
)

# 参数节点特殊处理：Parameter 的节点名 = param_name + 可选 value 后缀
def _parameter_node_name(param: dict) -> str:
    name = (param.get("param_name") or "").strip()
    value = (param.get("value") or "").strip()
    return f"{name}_{value}" if value and name else (name or param.get("name", ""))


class GraphStore:
    """全内嵌图谱存储：NetworkX MultiDiGraph + pickle 持久化"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        triples_dir: Optional[str] = None,
        graph_path: Optional[str] = None,
        exclude_fault_names: Optional[list] = None,
    ):
        self._lock = threading.RLock()
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "machine_fault.json"
        )
        self.triples_dir = triples_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "triples"
        )
        self.graph_path = graph_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "graph_cache.pkl"
        )
        # 构建图谱时过滤的故障名黑名单（用于剔除混入的脏数据，原始数据文件不动）
        self.exclude_fault_names = set(exclude_fault_names or [])
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._built = False

    # ---------- 节点/边原语 ----------

    def node_id(self, namespace: str, entity_type: str, name: str):
        return (namespace, entity_type, name)

    def has_node_id(self, nid) -> bool:
        return self.G.has_node(nid)

    def add_rule_node(self, entity_type: str, name: str, props: dict = None):
        """规则图谱节点：entity_type 必须是 RULE_NODE_TYPES 之一"""
        nid = self.node_id(NAMESPACE_RULE, entity_type, name)
        attrs = {"namespace": NAMESPACE_RULE, "entity_type": entity_type, "name": name}
        if props:
            attrs["props"] = dict(props)
        if not self.G.has_node(nid):
            self.G.add_node(nid, **attrs)
        elif props:
            self.G.nodes[nid].setdefault("props", {}).update(props)
        return nid

    def add_upload_node(self, entity_type: str, name: str, props: dict = None):
        nid = self.node_id(NAMESPACE_UPLOAD, entity_type, name)
        attrs = {"namespace": NAMESPACE_UPLOAD, "entity_type": entity_type, "name": name}
        if props:
            attrs["props"] = dict(props)
        if not self.G.has_node(nid):
            self.G.add_node(nid, **attrs)
        elif props:
            self.G.nodes[nid].setdefault("props", {}).update(props)
        return nid

    def add_edge(self, src_nid, dst_nid, rel_type: str, rel_name: str = None):
        self.G.add_edge(src_nid, dst_nid, rel_type=rel_type, rel_name=rel_name or rel_type)

    # ---------- 构建：规则图谱 ----------

    def _parse_list_or_numbered(self, value) -> list:
        """兼容 list 与 '1. xxx 2. yyy' 编号字符串两种格式"""
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            import re
            parts = re.split(r"\d+[\.、]", value)
            return [p.strip() for p in parts if p.strip()]
        return []

    @staticmethod
    def _iter_json_objects(path: str):
        """
        兼容两种数据格式：
        1. JSON 数组（[ {...}, {...} ]，单行或多行）
        2. JSONL（每行一个对象，允许行尾逗号）
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 尝试整体 JSON（数组格式）
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
                return
        except json.JSONDecodeError:
            pass
        # 回退：逐行解析（JSONL / 数组分行写但缺逗号 / 行尾带逗号）
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 兼容数组格式的分行写法：剥行首 '[' 与行尾 ']'、行尾逗号
            if line.startswith("["):
                line = line[1:].strip()
            if line.endswith("]"):
                line = line[:-1].strip()
            if line.endswith(","):
                line = line[:-1].strip()
            if not line or line in ("[", "]"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item

    def build_from_fault_json(self, path: Optional[str] = None) -> dict:
        """从 machine_fault.json 构建规则图谱（兼容数组/JSONL 两种格式），返回统计信息"""
        path = path or self.data_path
        stats = {"faults": 0, "nodes": 0, "edges": 0}
        with self._lock:
            for data in self._iter_json_objects(path):
                fault_name = (data.get("name") or "").strip()
                if not fault_name:
                    continue
                if fault_name in self.exclude_fault_names:
                    continue
                stats["faults"] += 1

                # FaultType 中心节点（含详细属性）
                fault_props = {k: data.get(k, "") for k in FAULT_TYPE_PROPS}
                fault_props["name"] = fault_name
                fault_nid = self.add_rule_node("FaultType", fault_name, fault_props)

                # 一对多关系：文本实体
                rel_map = [
                    ("cause", "Cause", "HAS_CAUSE"),
                    ("symptom", "Symptom", "HAS_SYMPTOM"),
                    ("solution", "Solution", "HAS_SOLUTION"),
                    ("component", "Component", "INVOLVES_COMPONENT"),
                    ("prevent", "Prevent", "HAS_PREVENT"),
                    ("check", "Check", "NEEDS_CHECK"),
                    ("material", "Material", "APPLIES_TO_MATERIAL"),
                    ("category", "Category", "BELONGS_TO_CATEGORY"),
                    ("cure_way", "CureWay", "HAS_CURE_WAY"),
                ]
                for key, ntype, rel in rel_map:
                    for item in self._parse_list_or_numbered(data.get(key, [])):
                        if len(item) <= 1:
                            continue
                        nid = self.add_rule_node(ntype, item)
                        self.add_edge(fault_nid, nid, rel)

                # Parameter：带 param_name/value/adjustment 属性
                param_list = data.get("parameter", [])
                if isinstance(param_list, str):
                    param_list = [{"param_name": param_list.strip()}]
                for p in param_list if isinstance(param_list, list) else []:
                    if not isinstance(p, dict):
                        continue
                    pname = _parameter_node_name(p)
                    if not pname:
                        continue
                    pnid = self.add_rule_node("Parameter", pname, {
                        "param_name": p.get("param_name", ""),
                        "value": p.get("value", ""),
                        "adjustment": p.get("adjustment", ""),
                    })
                    self.add_edge(fault_nid, pnid, "HAS_PARAMETER")

                stats["nodes"] = self.G.number_of_nodes()
                stats["edges"] = self.G.number_of_edges()
        return stats

    # ---------- 构建：人工导入三元组 ----------

    def add_triple(self, h: dict, r: dict, t: dict, namespace: str = NAMESPACE_UPLOAD):
        """添加一条三元组。h/t: {"name", "type"}；r: {"type"}"""
        h_name = (h.get("name") or "").strip()
        t_name = (t.get("name") or "").strip()
        r_type = (r.get("type") or "RELATED_TO").strip()
        if not h_name or not t_name:
            raise ValueError("三元组头/尾实体 name 不能为空")
        h_type = (h.get("type") or "Entity").strip() or "Entity"
        t_type = (t.get("type") or "Entity").strip() or "Entity"
        add_fn = self.add_upload_node if namespace == NAMESPACE_UPLOAD else self.add_rule_node
        h_nid = add_fn(h_type, h_name, {"type": h_type})
        t_nid = add_fn(t_type, t_name, {"type": t_type})
        self.add_edge(h_nid, t_nid, r_type, r_type)
        return h_nid, t_nid

    def load_triples(self, triples_dir: Optional[str] = None) -> dict:
        """加载 data/triples/ 中的人工三元组（.json / .jsonl，兼容数组 / JSONL 格式），返回统计信息"""
        triples_dir = triples_dir or self.triples_dir
        stats = {"files": 0, "triples": 0, "errors": []}
        if not os.path.isdir(triples_dir):
            return stats
        with self._lock:
            for filename in sorted(os.listdir(triples_dir)):
                if not (filename.endswith(".jsonl") or filename.endswith(".json")):
                    continue
                filepath = os.path.join(triples_dir, filename)
                stats["files"] += 1
                try:
                    for triple in self._iter_json_objects(filepath):
                        try:
                            self.add_triple(
                                triple.get("h", {}),
                                triple.get("r", {}),
                                triple.get("t", {}),
                            )
                            stats["triples"] += 1
                        except (ValueError, KeyError) as e:
                            stats["errors"].append(f"{filename}: {e}")
                except OSError as e:
                    stats["errors"].append(f"{filename}: {e}")
        return stats

    # ---------- 持久化 / 重建 ----------

    def rebuild(self) -> dict:
        """全量重建：清空 → 规则图谱 + 人工三元组 → 持久化缓存"""
        with self._lock:
            self.G = nx.MultiDiGraph()
            rule_stats = self.build_from_fault_json()
            triple_stats = self.load_triples()
            self._built = True
            self.save()
            return {
                "rule": rule_stats,
                "upload": triple_stats,
                "total_nodes": self.G.number_of_nodes(),
                "total_edges": self.G.number_of_edges(),
            }

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.graph_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.G, f)
        except OSError:
            pass  # 缓存失败不影响运行（可随时重建）

    def load(self, path: Optional[str] = None) -> bool:
        """从 pickle 缓存恢复图；失败返回 False"""
        path = path or self.graph_path
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                self.G = pickle.load(f)
            self._built = True
            return True
        except Exception:
            return False

    def ensure_built(self) -> None:
        """确保图谱已构建：优先用缓存，缓存不可用则重建"""
        if self._built and self.G.number_of_nodes() > 0:
            return
        if self.load():
            return
        self.rebuild()

    # ---------- 查询原语 ----------

    def iter_rule_nodes(self, entity_type: str):
        """遍历指定类型的规则节点，yield (node_id, attrs)"""
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("namespace") == NAMESPACE_RULE and attrs.get("entity_type") == entity_type:
                yield nid, attrs

    def find_nodes_by_name(self, name: str, namespace: Optional[str] = None) -> list:
        """精确名称匹配，返回 node_id 列表"""
        out = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("name") == name:
                if namespace is None or attrs.get("namespace") == namespace:
                    out.append(nid)
        return out

    def find_nodes_contains(self, keyword: str, namespace: Optional[str] = None, limit: int = 20) -> list:
        """子串匹配（CONTAINS 兜底），返回 node_id 列表（按名称长度升序）"""
        out = []
        for nid, attrs in self.G.nodes(data=True):
            if keyword in (attrs.get("name") or ""):
                if namespace is None or attrs.get("namespace") == namespace:
                    out.append(nid)
                    if len(out) >= limit:
                        break
        out.sort(key=lambda nid: len(self.G.nodes[nid].get("name", "")))
        return out

    def get_neighbors(self, nid, rel_types: Optional[list] = None) -> list:
        """直接邻居（出边），可选按 rel_type 过滤，返回 [(neighbor_nid, rel_type)]"""
        out = []
        for _, dst, edge_attrs in self.G.out_edges(nid, data=True):
            rt = edge_attrs.get("rel_type", "")
            if rel_types and rt not in rel_types:
                continue
            out.append((dst, rt))
        return out

    def expand_subgraph(
        self,
        seed_nids: list,
        hops: int = 2,
        rel_types: Optional[list] = None,
        exclude_rel_types: Optional[set] = None,
    ) -> dict:
        """
        多跳子图扩展（BFS），返回 {"nodes": [...], "edges": [...]}
        nodes: {"id", "name", "type", "namespace", "props"}
        edges: {"source": name, "target": name, "type", "source_id", "target_id"}
        exclude_rel_types: 跳过这些关系类型的边（及其更深层扩展）。用于剔除
            "使用工具/需要工具"等不承载诊断因果语义的工具关系，避免 2 跳扩展
            把与故障本身无关的实体（红外热像仪、轴承拉拔器等）卷进结果。
        """
        nodes = {}
        edges = {}
        frontier = set(seed_nids)
        visited = set(seed_nids)
        for nid in seed_nids:
            if self.G.has_node(nid):
                nodes[nid] = self._node_to_dict(nid)
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                if not self.G.has_node(nid):
                    continue
                for dst, rt in self.get_neighbors(nid, rel_types):
                    if exclude_rel_types and rt in exclude_rel_types:
                        continue
                    if dst not in nodes:
                        nodes[dst] = self._node_to_dict(dst)
                    ekey = (nid, dst, rt)
                    if ekey not in edges:
                        edges[ekey] = {
                            "source_id": nid,
                            "target_id": dst,
                            "type": rt,
                            "source": self.G.nodes[nid].get("name", ""),
                            "target": self.G.nodes[dst].get("name", ""),
                        }
                    if dst not in visited:
                        visited.add(dst)
                        next_frontier.add(dst)
            frontier = next_frontier
            if not frontier:
                break
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    def _node_to_dict(self, nid) -> dict:
        attrs = self.G.nodes[nid]
        return {
            "id": nid,
            "name": attrs.get("name", ""),
            "type": attrs.get("entity_type", ""),
            "namespace": attrs.get("namespace", ""),
            "props": attrs.get("props", {}),
        }

    def triples_from_subgraph(self, subgraph: dict, seed_names: set) -> list:
        """
        将子图转换为标准三元组列表。
        {"h": {"name","type"}, "r": {"type"}, "t": {"name","type"}}
        以种子实体为中心的 1 跳边优先转换为三元组；2 跳边保留（用于图结构展示）。
        """
        triples = []
        for e in subgraph["edges"]:
            triples.append({
                "h": {"name": e["source"], "type": self._type_of(e["source_id"])},
                "r": {"type": e["type"]},
                "t": {"name": e["target"], "type": self._type_of(e["target_id"])},
            })
        return triples

    def _type_of(self, nid) -> str:
        return self.G.nodes[nid].get("entity_type", "") if self.G.has_node(nid) else ""

    def stats(self) -> dict:
        """图谱统计信息"""
        ns_stats = {}
        type_stats = {}
        for nid, attrs in self.G.nodes(data=True):
            ns = attrs.get("namespace", "")
            ns_stats[ns] = ns_stats.get(ns, 0) + 1
            t = attrs.get("entity_type", "")
            key = f"{ns}:{t}"
            type_stats[key] = type_stats.get(key, 0) + 1
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "namespace_stats": ns_stats,
            "type_stats": type_stats,
        }

    def all_entity_names(self, namespace: Optional[str] = None) -> list:
        """全部实体名（用于 AC 自动机 / 向量索引）"""
        out = []
        for nid, attrs in self.G.nodes(data=True):
            if namespace is None or attrs.get("namespace") == namespace:
                out.append(attrs.get("name", ""))
        return [n for n in out if n]
