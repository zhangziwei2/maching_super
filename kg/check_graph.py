from py2neo import Graph

g = Graph("bolt://localhost:7687", auth=("neo4j", "200980216"))
# 检查所有节点数量和标签
result = g.run("MATCH (n) RETURN labels(n) AS label, count(n) AS count")
print("图谱节点统计:", result.data())

# 检查所有关系数量
result2 = g.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count")
print("图谱关系统计:", result2.data())
