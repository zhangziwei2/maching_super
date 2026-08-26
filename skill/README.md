# 颤振诊断 Skill

对上传的传感器 CSV 信号进行加工颤振诊断（**双模式**，v3.1.0）：①颤振诊断模式，输出三分类 **稳定 / 轻微颤振 / 严重颤振**；②实时监控模式，基于稳态基线做 z-score 偏离监测（🟢<2.0 / 🟡2.0~3.5 / 🔴≥3.5）。诊断与监控均会在生成报告前先输出 **4×1 原始信号时序图**（`<csv_stem>_signal.png`）。

流程：50 维特征提取 → 归一化 → SAE 降维 16 维 → 弱分类器 → Stacking 集成。

## 快速开始

```bash
# 1. 安装依赖
cd script
pip install -r requirements.txt

# 2. 训练（首次使用或数据更新后；可选指定训练 CSV）
python -m script.train_all [训练数据.csv]

# 3. 诊断
python -m script.chatter_diagnosis_skill signal.csv
# 或 Python 调用
python -c "from script.chatter_diagnosis_skill import diagnose_csv; print(diagnose_csv('signal.csv'))"
```

> **实时监控模式**（可选）：首次或数据更新后，额外运行 `python -m script.train_fusion_model` 生成 `script/models/baseline_stats.json`（稳态基线），即可对在线信号做逐段 z-score 偏离监测。详见 `SKILL.md` / `reference/README.md`。

## CSV 输入格式（诊断用）

| 列序 | 内容 |
|------|------|
| 1 | 时间（秒） |
| 2 | X 轴振动 |
| 3 | Y 轴振动 |
| 4 | Z 轴振动 |
| 5 | 三向力合力（N） |

采样点要求：**≥ 256 点**（不足一个分段 256 点无法诊断）。

## 文档导航
- `SKILL.md` — 完整技能定义与使用说明（**权威**）
- `reference/README.md` — 技术参考手册（特征 / 模型 / 参数）
- `reference/chatter_diagnosis.md` — 旧版 (v1.0.0) 定义（历史归档）
