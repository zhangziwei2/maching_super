import sys
sys.path.insert(0, "backend")
from backend.graphkb import get_graph_kb

kb = get_graph_kb()
print("=== source status ===")
print(kb.source_status())
print()
print("=== domain query: 主轴过热是什么原因 ===")
print(kb.query_text("主轴过热是什么原因"))
print()
print("=== manual_triples query (Neo4j down -> degrade empty) ===")
print(repr(kb.query_text("颤振")))
print()
print("=== structured domain query ===")
r = kb.query_structured("刀具磨损", hops=2, top_k=5)
print("nodes:", len(r.get("nodes", [])), "edges:", len(r.get("edges", [])), "triples:", len(r.get("triples", [])))
