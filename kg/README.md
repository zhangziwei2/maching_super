# 机床故障诊断 GraphRAG 知识图谱问答系统

基于医疗知识图谱项目（QAMedicalKG）重构的机床铣削/车削故障诊断智能问答系统。

> **v2.0 架构升级（2026-08）**：移除 Neo4j/py2neo 依赖，切换为**全内嵌 NetworkX 图引擎**。
> 新增模块：`graph_store.py`（图存储+持久化）、`graph_service.py`（混合检索服务）、
> `graph_adapter.py`（GraphAdapter 抽象层）、`graph_schema.py`（Schema 定义）、
> `triple_importer.py`（人工三元组导入 + LLM 校验）。详见 `../docs/KNOWLEDGE_GRAPH_UPGRADE.md`。
> 原有 `build_machinegraph.py`（Neo4j 版）保留仅作参考，不再参与运行。

## 项目结构

```
maching/
├── data/
│   ├── machine_fault.json          # 机床故障知识数据（13种核心故障类型）
│   ├── triples/                    # [新增] 人工/专家导入的三元组 JSONL
│   └── graph_cache.pkl             # [新增] NetworkX 图 pickle 缓存（可重建）
├── dict/                             # 实体词典
│   ├── fault_type.txt                # 故障类型词典
│   ├── phenomenon.txt                # 现象词典
│   ├── component.txt                 # 部件词典
│   ├── solution.txt                  # 解决方法词典
│   ├── detection.txt                 # 检测方法词典
│   ├── cause.txt                     # 故障原因词典
│   ├── material.txt                  # 加工材料词典
│   ├── parameter.txt                 # 加工参数词典
│   └── deny.txt                      # 否定词词典
├── prepare_data/
│   └── __init__.py
├── graph_schema.py                  # [新增] 实体/关系 Schema 与命名空间定义
├── graph_store.py                   # [新增] NetworkX 图引擎（构建/持久化/子图扩展）
├── graph_adapter.py                 # [新增] GraphAdapter 抽象层（Rule/Upload 适配器）
├── graph_service.py                 # [新增] 统一混合检索服务（实体锚定+子图扩展→三元组）
├── triple_importer.py               # [新增] JSONL 三元组导入 + LLM 一致性校验
├── question_classifier.py            # 意图识别（AC自动机）
├── question_parser.py                # [改造] 生成结构化查询指令（替代模板 Cypher）
├── answer_search.py                  # [改造] NetworkX 图遍历执行 + 答案格式化
├── chatbot_graph.py                  # 主入口（NetworkX 版）
├── sensor_interface.py               # 传感器实时数据接口（预留）
├── tavily_search.py                  # Tavily联网搜索模块
├── graphrag_enhancer.py              # [改造] 图谱增强检索（语义检索已落地）
├── config.py                         # 全局配置文件（去 Neo4j，新增 GRAPH_CONFIG）
└── requirements.txt                  # Python依赖（py2neo → networkx）
```

## 支持的7种核心问答意图

| 意图类型 | 触发条件 | 示例问句 |
|---------|---------|---------|
| 故障原因查询 | fault_type + cause_qwds | "主轴过热是什么原因？" |
| 现象诊断 | phenomenon + fault_type_qwds | "铣削时出现振纹是什么故障？" |
| 解决方法推荐 | fault_type + solution_qwds | "刀具崩刃怎么处理？" |
| 部件关联故障 | component + fault_type_qwds | "导轨一般会出现什么故障？" |
| 预防措施 | fault_type + prevent_qwds | "如何避免刀具磨损？" |
| 参数优化 | phenomenon + parameter_qwds | "表面粗糙度差该怎么调参数？" |
| 检测手段 | fault_type + detection_qwds | "怎么检测主轴是否过热？" |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- `networkx` - 图引擎（v2.0 替代 py2neo/Neo4j）
- `pyahocorasick` - AC自动机（快速多模式匹配）
- `requests` - HTTP请求（Tavily搜索）
- `numpy`, `scipy`, `scikit-learn` - 数据分析
- `websockets` - WebSocket传感器数据流

### 2. 构建知识图谱（全内嵌，无需任何外部数据库）

```bash
python -c "from graph_service import graph_service; print(graph_service.rebuild())"
```

首次运行会从 `data/machine_fault.json` 构建规则图谱，并加载 `data/triples/*.jsonl`
人工三元组，然后缓存到 `data/graph_cache.pkl`。**无需安装/启动 Neo4j**。

### 4. 运行问答系统

```bash
python chatbot_graph.py
```

交互模式示例：
```
咨询: 主轴过热是什么原因？
客服机器人: 【主轴过热】的可能原因包括：...

咨询: sensor: 振动过大怎么回事？
客服机器人: 【振动过大】的可能原因包括：...

[传感器实时数据辅助诊断]
传感器数据正常。当前振动RMS: 2.50 m/s²...
```

### 5. 导入人工三元组（专家补充）

将 JSON / JSONL 文件放入 `data/triples/` 目录（兼容 JSON 数组、`{"triples":[...]}` 包装、
逐行 JSONL 三种格式，字段见 `triple_importer.py`），或执行：

```bash
python triple_importer.py data/triples/your_file.json
```

导入后立即生效，重启不丢失；删除该文件后重建图谱即可整体回滚。

## 核心模块说明

### 意图识别 (question_classifier.py)

基于 `ahocorasick` AC自动机进行实体识别，结合规则匹配疑问词，支持以下实体类型：
- `fault_type` - 故障类型（刀具磨损、崩刃、主轴过热等）
- `phenomenon` - 现象（振动、异响、表面振纹等）
- `component` - 部件（主轴、导轨、刀库等）
- `solution` - 解决方法
- `detection` - 检测方法
- `cause` - 故障原因
- `material` - 加工材料
- `parameter` - 加工参数

### 知识图谱 Schema (Neo4j)

**节点类型 (6类):**
- `FaultType` - 故障类型（含desc, cause, prevent, parameter等属性）
- `Component` - 机床部件
- `Phenomenon` - 故障现象
- `Solution` - 解决方法
- `Detection` - 检测方法
- `Material` - 加工材料

**关系类型 (5种):**
- `FaultType -[has_symptom]-> Phenomenon`
- `FaultType -[involves_component]-> Component`
- `FaultType -[has_solution]-> Solution`
- `FaultType -[need_check]-> Detection`
- `FaultType -[applies_to]-> Material`

### 传感器接口 (sensor_interface.py)

预留接口，支持接入：
- 3通道振动传感器（X/Y/Z方向）
- 3通道力传感器（Fx/Fy/Fz）
- 1通道声级计（SPL）

配置项见 `config.py` 中的 `SENSOR_CONFIG`。

### Tavily联网搜索 (tavily_search.py)

当知识图谱无匹配结果时，自动通过Tavily API从互联网搜索答案。

配置项：
```python
TAVILY_CONFIG = {
    "api_key": "ly-dev-482SnJ-wkW31Zq4gHI8PNM8d9ypOzKA3FXZPcFwjJaNxUuCUa",  # 替换为实际API Key
    "max_results": 5,
    "search_depth": "advanced",
}
```

### GraphRAG增强 (graphrag_enhancer.py)

将知识图谱数据转换为文本块，支持基于关键词的语义增强检索。
可扩展为向量数据库（Milvus/Pinecone/Chroma）实现真正的语义检索。

## 数据覆盖

当前 `machine_fault.json` 包含12种核心机床故障类型：

1. 刀具磨损 - 最常见故障，渐进性损伤
2. 崩刃 - 突发性刀具损伤
3. 主轴过热 - 主轴系统温度异常
4. 导轨爬行 - 低速运动不平稳
5. 丝杠间隙 - 进给系统精度下降
6. 轴承损坏 - 旋转部件核心故障
7. 伺服报警 - 伺服系统保护性提示
8. 振动过大 - 包含颤振和强迫振动
9. 表面粗糙度差 - 加工质量缺陷
10. 尺寸超差 - 精度故障
11. 异响 - 异常声音预兆
12. 刀库换刀故障 - 加工中心特有故障
13. 数控系统报警 - 控制系统异常

每种故障包含：故障描述、原因、症状、涉及部件、解决方法、检测方法、预防措施、参数建议、适用材料、修复时间、修复概率等完整信息。

## 扩展指南

### 添加新故障类型

1. 在 `data/machine_fault.json` 中追加新故障JSON行
2. 运行 `build_machinegraph.py` 重建图谱
3. 或直接使用Neo4j Cypher语句插入新节点和关系

### 接入真实传感器

1. 在 `config.py` 中设置 `SENSOR_CONFIG["enabled"] = True`
2. 配置WebSocket或HTTP端点地址
3. 实现传感器数据推送逻辑，调用 `SensorInterface.push_data()`

### 启用向量语义检索

1. 选择向量数据库（推荐Chroma/Milvus）
2. 在 `graphrag_enhancer.py` 中实现 `semantic_search()` 方法
3. 生成文本块嵌入向量并入库

## 与医疗知识图谱的对比

| 维度 | 医疗系统 (QAMedicalKG) | 机床故障系统 (本项目) |
|------|----------------------|---------------------|
| 领域 | 疾病诊疗 | 机床故障诊断 |
| 核心节点 | Disease | FaultType |
| 关系数 | 10+种 | 5种（精简聚焦） |
| 实体类型 | 7类 | 6类 + 参数/材料 |
| 扩展功能 | 无 | 传感器接口、Tavily、GraphRAG |
| 应用场景 | 在线问诊 | 智能制造/工业诊断 |

## License

本项目基于开源医疗知识图谱项目重构，仅供学习和研究使用。
