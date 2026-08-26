# 机床故障诊断GraphRAG知识图谱问答系统 - 全字段建模版本

## 📺 项目简介

本项目是一个基于知识图谱的机床故障诊断问答系统，使用Neo4j图数据库存储故障诊断知识，支持自然语言问答、GraphRAG增强、Tavily联网搜索兜底、传感器实时数据接口等功能。

### ✨ 全字段建模特性

本版本对原系统进行了全字段建模升级：

1. **所有JSON字段都转化为节点或属性**
   - 10种节点类型：FaultType, Symptom, Component, Solution, Check, Material, Cause, Prevent, Parameter, CureWay, Category
   - 所有字段（desc, cause, prevent, parameter, cure_way, easy_get, cure_lasttime, cured_prob）都建模为节点或属性

2. **8种关系类型，支持多跳查询**
   - HAS_SYMPTOM - 故障有症状
   - INVOLVES_COMPONENT - 故障涉及部件
   - HAS_SOLUTION - 故障有解决方案
   - NEEDS_CHECK - 故障需要检测
   - APPLIES_TO_MATERIAL - 故障适用于材料
   - HAS_CAUSE - 故障有原因
   - HAS_PREVENT - 故障有预防措施
   - BELONGS_TO_CATEGORY - 故障属于类别

3. **新增查询类型，覆盖所有业务场景**
   - 20种意图类型，覆盖故障诊断的各个方面
   - 支持多跳查询：症状→故障→原因→解决方案

4. **综合查询：fault_full_info可一次性获取故障的所有信息**
   - 一次性获取故障的所有相关信息（症状、原因、预防、参数、部件、解决方案、检测方法、材料、类别、修复方式等）

5. **数据导入：配套的Neo4j导入脚本，一键导入JSON数据**
   - 一键导入脚本 `import_knowledge_graph.py`
   - 自动清空数据库、构建图谱、导出数据、验证导入结果

## 📂 项目结构

```
maching/
├── data/
│   └── machine_fault.json        # 故障数据（13种故障类型）
├── dict/                         # 实体词典
│   ├── fault_type.txt            # 故障类型词典
│   ├── phenomen.py.txt           # 现象词典
│   ├── component.txt            # 部件词典
│   ├── solution.txt             # 解决方法词典
│   ├── detection.txt            # 检测方法词典
│   ├── cause.txt               # 原因词典
│   ├── prevent.txt             # 预防措施词典
│   ├── parameter.txt           # 参数优化词典
│   ├── cure_way.txt           # 修复方式词典
│   ├── category.txt            # 故障类别词典
│   ├── material.txt            # 材料词典
│   └── deny.txt               # 否定词词典
├── build_machinegraph.py       # 图数据库构建脚本（简单版本）
├── build_machinegraph_full.py  # 图数据库构建脚本（全字段建模版本）✅
├── import_knowledge_graph.py   # 一键导入脚本✅
├── question_classifier.py      # 意图识别/问题分类模块✅
├── question_parser.py          # 问题解析器（Cypher SQL生成模块）✅
├── answer_search.py            # 答案搜索与格式化模块✅
├── chatbot_graph.py            # 主入口文件✅
├── test_system.py              # 测试脚本（简单版本）
├── test_system_full.py        # 测试脚本（全字段建模版本）✅
├── config.py                  # 配置文件
├── tavily_search.py           # Tavily联网搜索模块
├── graphrag_enhancer.py      # GraphRAG增强模块
├── sensor_interface.py         # 传感器接口模块
└── README.md                 # 使用说明✅
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install py2neo pyahocorasick requests numpy websockets asyncio-mqtt scipy scikit-learn
```

### 2. 启动Neo4j数据库

确保Neo4j数据库已启动，默认连接参数：
- URI: `bolt://localhost:7687`
- 用户名: `neo4j`
- 密码: `200980216`（请修改为你的真实密码）

### 3. 一键构建知识图谱

```bash
python import_knowledge_graph.py
```

这个脚本会：
1. 检查Neo4j连接
2. 清空现有数据库
3. 构建全字段知识图谱（10种节点类型，8种关系类型）
4. 导出数据到dict目录
5. 验证导入结果
6. 测试样本查询

### 4. 启动聊天机器人

```bash
python chatbot_graph.py
```

然后就可以输入问题了，例如：
- `刀具磨损是什么原因？`
- `铣削时出现振纹是什么故障？`
- `刀具崩刃怎么处理？`
- `详细介绍刀具磨损的所有信息`（综合查询）

### 5. 运行测试脚本

```bash
python test_system_full.py
```

这个脚本会测试所有功能：
- 数据文件完整性
- 字典文件完整性
- 配置文件
- 意图识别（20种意图类型）
- 查询生成（20种查询类型）
- 答案搜索
- 综合查询（fault_full_info）
- 多跳查询
- Neo4j连接
- 图数据库查询

## 🔧 配置说明

配置文件 `config.py` 包含以下配置：

```python
# Neo4j配置
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "200980216"  # 改成你的真实密码
}

# Tavily API配置
TAVILY_CONFIG = {
    "api_key": "tvly-your-api-key-here",  # 改成你的Tavily API Key
    "search_depth": "advanced",
    "max_results": 5
}

# 传感器配置
SENSOR_CONFIG = {
    "enabled": False,
    "vibration_channels": 3,  # 3轴振动
    "force_channels": 3,         # 3轴力
    "sound_level": True,          # 声级计
    "alarm_threshold": {
        "vibration": 5.0,       # 振动报警阈值 (g)
        "force": 1000,           # 力报警阈值 (N)
        "sound": 85               # 声级报警阈值 (dB)
    }
}

# GraphRAG配置
GRAPHRAG_CONFIG = {
    "enabled": True,
    "chunk_size": 500,
    "overlap": 50
}

# 聊天机器人配置
CHATBOT_CONFIG = {
    "enable_tavily_fallback": True,
    "num_limit": 20,
    "default_answer": "您好，我是机床故障诊断助手。请问您遇到了什么故障问题？"
}
```

## 📚 使用指南

### 支持的意图类型（20种）

1. **故障原因查询** - `fault_cause`
   - 示例：`主轴过热是什么原因？`

2. **现象诊断** - `phenomenon_diagnosis`
   - 示例：`铣削时出现振纹是什么故障？`

3. **解决方法推荐** - `fault_solution`
   - 示例：`刀具崩刃怎么处理？`

4. **部件关联故障** - `component_fault`
   - 示例：`导轨一般会出现什么故障？`

5. **预防措施** - `fault_prevent`
   - 示例：`如何避免刀具磨损？`

6. **参数优化** - `parameter_optimize` / `fault_parameter`
   - 示例：`表面粗糙度差该怎么调参数？`

7. **检测手段** - `fault_detection` / `phenomenon_detection`
   - 示例：`怎么检测主轴是否过热？`

8. **故障类别查询** - `fault_category`
   - 示例：`刀具磨损属于什么类别？`

9. **修复方式查询** - `fault_cure_way`
   - 示例：`主轴过热该怎么修复？`

10. **易发情况查询** - `fault_easy_get`
    - 示例：`刀具磨损在什么情况下容易发生？`

11. **修复时间查询** - `fault_cure_lasttime`
    - 示例：`主轴过热修复需要多长时间？`

12. **修复概率查询** - `fault_cured_prob`
    - 示例：`刀具磨损能修好吗？`

13. **综合查询（故障全信息）** - `fault_full_info`
    - 示例：`详细介绍刀具磨损的所有信息`

14. **多跳查询：症状→故障→原因→解决方案** - `symptom_to_solution`

15. **多跳查询：部件→故障→检测方法** - `component_to_check`

16. **多跳查询：材料→故障→预防措施** - `material_to_prevent`

17. **现象→故障** - `phenomenon_fault`
    - 示例：`加工时出现异响是怎么回事？`

18. **材料相关故障** - `material_fault`
    - 示例：`钛合金加工容易出现什么问题？`

19. **故障描述** - `fault_desc`
    - 示例：`介绍一下刀具磨损`

20. **传感器辅助诊断** - `sensor:` 前缀
    - 示例：`sensor: 主轴过热是什么原因？`

### 综合查询示例

综合查询 `fault_full_info` 可以一次性获取故障的所有信息：

```
Q: 详细介绍刀具磨损的所有信息

A: 【刀具磨损】完整信息：

📝 描述：刀具磨损是数控加工中最常见的故障类型之一，指刀具在切削过程中由于机械、热和化学作用导致切削刃钝化或材料损失...
🏷️  类别：刀具故障，渐进性故障
🔍 症状：切削力增大，切削温度升高，加工表面粗糙度变差，切屑颜色异常...
⚠️ 原因：1. 切削速度过高导致刀具温度过高 2. 进给量过大加剧刀具与工件摩擦...
🛡️ 预防：1. 合理选择刀具材料和涂层 2. 控制切削速度和进给量在推荐范围内...
⚙️ 参数建议：切削速度vc: 80-150m/min(钢件); 进给量f: 0.1-0.3mm/r...
🔧 涉及部件：刀具，刀柄，主轴
🔨 解决方法：更换新刀具，降低切削速度20%-30%，减小进给量...
🔧 修复方式：机械修复，参数调整，刀具更换
🔬 检测方法：目视检查刀具刃口，测量刀具后刀面磨损量VB...
🧪 适用材料：钢件，铸铁，铝合金，钛合金，不锈钢，高温合金
⏰ 易发情况：长时间连续加工、高速切削、难加工材料（如钛合金、高温合金、淬硬钢）加工时更易发生
⏱️  修复时间：取决于刀具材质和切削参数，硬质合金刀具一般可连续加工2-8小时
📊 修复概率：通过参数优化和刀具管理，刀具寿命可延长30%-50%
```

### 多跳查询示例

多跳查询支持跨多个节点的复杂查询：

```
Q: 切削力增大是怎么回事？

A: 出现【切削力增大】现象的诊断与解决方案：

可能故障：刀具磨损，崩刃，主轴过热，轴承损坏，伺服报警，振动过大，尺寸超差
可能原因：1. 切削速度过高导致刀具温度过高 2. 进给量过大加剧刀具与工件摩擦...
推荐解决方案：更换新刀具，降低切削速度20%-30%，减小进给量...
```

## 🔬 技术细节

### 全字段建模

本版本对原系统进行了全字段建模升级，将所有JSON字段都转化为节点或属性：

1. **节点类型（10种）**：
   - `FaultType` - 故障类型（中心节点）
   - `Symptom` - 症状/现象
   - `Component` - 部件
   - `Solution` - 解决方案
   - `Check` - 检测方法
   - `Material` - 加工材料
   - `Cause` - 故障原因
   - `Prevent` - 预防措施
   - `Parameter` - 参数优化
   - `CureWay` - 修复方式
   - `Category` - 故障类别

2. **关系类型（8种）**：
   - `HAS_SYMPTOM` - 故障有症状
   - `INVOLVES_COMPONENT` - 故障涉及部件
   - `HAS_SOLUTION` - 故障有解决方案
   - `NEEDS_CHECK` - 故障需要检测
   - `APPLIES_TO_MATERIAL` - 故障适用于材料
   - `HAS_CAUSE` - 故障有原因
   - `HAS_PREVENT` - 故障有预防措施
   - `BELONGS_TO_CATEGORY` - 故障属于类别

3. **属性字段**：
   - `FaultType.desc` - 故障描述
   - `FaultType.easy_get` - 易发情况
   - `FaultType.cure_lasttime` - 修复时间
   - `FaultType.cured_prob` - 修复概率

### 多跳查询

多跳查询支持跨多个节点的复杂查询，例如：

- **症状→故障→原因→解决方案**：
  ```
  MATCH (f:FaultType)-[:HAS_SYMPTOM]->(s:Symptom {name: '切削力增大'})
  OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
  OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
  RETURN s.name AS symptom, f.name AS fault,
         collect(DISTINCT c.name) AS causes,
         collect(DISTINCT sol.name) AS solutions
  ```

- **部件→故障→检测方法**：
  ```
  MATCH (f:FaultType)-[:INVOLVES_COMPONENT]->(c:Component {name: '刀具'})
  OPTIONAL MATCH (f)-[:NEEDS_CHECK]->(chk:Check)
  RETURN c.name AS component, f.name AS fault,
         collect(DISTINCT chk.name) AS checks
  ```

- **材料→故障→预防措施**：
  ```
  MATCH (f:FaultType)-[:APPLIES_TO_MATERIAL]->(m:Material {name: '钛合金'})
  OPTIONAL MATCH (f)-[:HAS_PREVENT]->(p:Prevent)
  RETURN m.name AS material, f.name AS fault,
         collect(DISTINCT p.name) AS preventions
  ```

### 综合查询

综合查询 `fault_full_info` 可以一次性获取故障的所有信息：

```
MATCH (f:FaultType {name: '刀具磨损'})
OPTIONAL MATCH (f)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (f)-[:INVOLVES_COMPONENT]->(c:Component)
OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)
OPTIONAL MATCH (f)-[:NEEDS_CHECK]->(chk:Check)
OPTIONAL MATCH (f)-[:APPLIES_TO_MATERIAL]->(m:Material)
OPTIONAL MATCH (f)-[:BELONGS_TO_CATEGORY]->(cat:Category)
OPTIONAL MATCH (f)-[:HAS_CURE_WAY]->(cw:CureWay)
OPTIONAL MATCH (f)-[:HAS_CAUSE]->(cau:Cause)
OPTIONAL MATCH (f)-[:HAS_PREVENT]->(pre:Prevent)
OPTIONAL MATCH (f)-[:HAS_PARAMETER]->(param:Parameter)
RETURN 
    f.name AS fault,
    f.desc AS description,
    f.easy_get AS easy_get,
    f.cure_lasttime AS cure_time,
    f.cured_prob AS cure_probability,
    collect(DISTINCT s.name) AS symptoms,
    collect(DISTINCT c.name) AS components,
    collect(DISTINCT sol.name) AS solutions,
    collect(DISTINCT chk.name) AS checks,
    collect(DISTINCT m.name) AS materials,
    collect(DISTINCT cat.name) AS categories,
    collect(DISTINCT cw.name) AS cure_ways,
    collect(DISTINCT cau.name) AS causes,
    collect(DISTINCT pre.name) AS preventions,
    collect(DISTINCT param.name) AS parameters
```

## 🔧 扩展开发

### 添加新的故障类型

1. 在 `data/machine_fault.json` 中添加新的故障记录
2. 运行 `python import_knowledge_graph.py` 重新构建知识图谱
3. 更新 `dict/` 目录下的词典文件（可选，会自动导出）

### 添加新的意图类型

1. 在 `question_classifier.py` 中添加新的疑问词和分类逻辑
2. 在 `question_parser.py` 中添加新的查询生成逻辑
3. 在 `answer_search.py` 中添加新的答案格式化逻辑
4. 运行 `python test_system_full.py` 测试

### 集成传感器数据

1. 修改 `sensor_interface.py` 中的传感器配置
2. 启动传感器数据模拟器或接入真实传感器
3. 使用 `sensor:` 前缀进行传感器辅助诊断

## 📁 数据格式

### 故障数据格式（machine_fault.json）

```json
{
  "name": "刀具磨损",
  "desc": "刀具磨损是数控加工中最常见的故障类型之一...",
  "category": ["刀具故障", "渐进性故障"],
  "prevent": "1. 合理选择刀具材料和涂层\n2. 控制切削速度和进给量在推荐范围内...",
  "cause": "1. 切削速度过高导致刀具温度过高\n2. 进给量过大加剧刀具与工件摩擦...",
  "symptom": ["切削力增大", "切削温度升高", "加工表面粗糙度变差"],
  "component": ["刀具", "刀柄", "主轴"],
  "solution": ["更换新刀具", "降低切削速度20%-30%"],
  "check": ["目视检查刀具刃口", "测量刀具后刀面磨损量VB"],
  "cure_way": ["机械修复", "参数调整", "刀具更换"],
  "easy_get": "长时间连续加工、高速切削、难加工材料...",
  "parameter": "切削速度vc: 80-150m/min(钢件); 进给量f: 0.1-0.3mm/r...",
  "material": ["钢件", "铸铁", "铝合金", "钛合金"],
  "cure_lasttime": "取决于刀具材质和切削参数...",
  "cured_prob": "通过参数优化和刀具管理，刀具寿命可延长30%-50%"
}
```

## 🔍 故障排查

### Neo4j连接失败

1. 检查Neo4j是否已启动
2. 检查URI、用户名、密码是否正确
3. 检查防火墙设置

### 查询结果不正确

1. 检查意图识别是否正确
2. 检查词典文件是否完整
3. 检查Neo4j中是否有数据

### 综合查询返回结果不完整

1. 检查关系中是否有数据
2. 使用Neo4j Browser查看图数据结构
3. 运行 `python test_system_full.py` 测试

## 📚 参考资料

- [Neo4j官方文档](https://neo4j.com/docs/)
- [py2neo文档](https://py2neo.readthedocs.io/)
- [Cypher查询语言参考](https://neo4j.com/docs/cypher-manual/current/)
- [Tavily API文档](https://tavily.com/docs)
- [GraphRAG论文](https://arxiv.org/abs/2404.16130)

## 📝 更新日志

### v2.0.0 (全字段建模版本)

- ✅ 全字段建模：所有JSON字段都转化为节点或属性
- ✅ 8种关系类型，支持多跳查询
- ✅ 新增查询类型，覆盖所有业务场景
- ✅ 综合查询：fault_full_info可一次性获取故障的所有信息
- ✅ 数据导入：配套的Neo4j导入脚本，一键导入JSON数据
- ✅ 20种意图类型
- ✅ 多跳查询支持
- ✅ 完善的测试脚本

### v1.0.0 (初始版本)

- 初始版本，基于QAMedicalKG重构
- 7种核心意图
- 5种节点类型
- 5种关系类型
