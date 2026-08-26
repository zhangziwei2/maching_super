from py2neo import Graph

g = Graph("bolt://localhost:7687", auth=("neo4j", "200980216"))
g.run("MATCH (n) DETACH DELETE n")
print("Neo4j 数据库已清空")
