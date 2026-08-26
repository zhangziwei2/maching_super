---
name: chatter-diagnosis
displayName: 颤振诊断
description: 颤振诊断双模式 Skill：①诊断模式（50维特征提取 → 归一化 → SAE 降维16维 → 弱分类器 → Stacking 集成分类，输出三分类稳定/轻微颤振/严重颤振）；②实时监控模式（基于稳态基线的 z-score 偏离监测）。
version: 3.1.0
author: siat
type: diagnosis
input:
  - name: csv_path
    type: string
    required: true
    description: 传感器 CSV 文件路径（时间, X振动, Y振动, Z振动, 三向力合力）
output:
  - name: report
    type: string
    description: 结构化诊断报告（含 4×1 原始信号图路径）
  - name: signal_plot
    type: image
    description: 4×1 原始信号时序图 PNG（<csv_stem>_signal.png），与报告同步生成
dependencies:
  - python>=3.9
  - numpy>=1.21
  - pandas>=1.3
  - scipy
  - torch>=1.9
  - scikit-learn>=1.0
  - lightgbm
  - catboost
tags: [颤振, 加工状态监测, 信号处理, 故障诊断, SAE, Stacking]
---

# 颤振诊断 Skill

## 简介

本 Skill 提供**双模式**：对上传的传感器 CSV 信号，既可做颤振诊断（三分类），也可做实时 z-score 基线监控（见下文「实时监控模式」）。

| 类别 | 编号 | 含义 |
|------|------|------|
| 稳定 | 0 | 加工状态正常 |
| 轻微颤振 | 1 | 出现早期颤振征兆 |
| 严重颤振 | 2 | 颤振已充分发展，需立即干预 |

方法来源（与 senordata 参考代码严格一致）：
- **降维**：`2.2_SAEDimensionreduction.py` — StandardScaler 归一化 + 稀疏自编码器（SAE）降维
- **分类**：`3.1_Decision_Integration.py` — 弱分类器 + F1 自适应加权 + Stacking 集成

## 诊断模式流程（SAE + Stacking，唯一诊断管线）

```
传感器 CSV (时间, X振动, Y振动, Z振动, 三向力合力)
    ↓
[读取即绘制 4×1 信号图 <csv_stem>_signal.png]   ← v3.1 新增，不依赖模型
    ↓
分段 (256点/段，与训练一致)
    ↓
特征提取: 15×3 + 5 = 50 维
    ↓
StandardScaler 归一化 (scaler.pkl)
    ↓
SAE 稀疏自编码器降维: 50 → 512 → 256 → 128 → 16
    ↓
16 维编码特征再标准化 (scaler16.pkl)
    ↓
弱分类器: LightGBM / CatBoost / SVM / ExtraTrees / KNN / MLP
    ├─ F1 自适应加权投票（参考结果）
    ↓
Stacking 集成 (LR 元学习器, cv=5, predict_proba, passthrough)
    ↓
[稳定 / 轻微颤振 / 严重颤振]
```

## 信号时序图（v3.1 新增）

在读入 CSV 并校验通过后、特征提取之前，自动绘制 **4×1 原始信号时序图**，保存为
`<csv_stem>_signal.png`（与输入 CSV 同目录）。该图**不依赖任何诊断模型**——
即使模型文件缺失也能正常出图，用于在阅读诊断结论前先直观看到原始信号形态。

| 子图行 | 通道 | 对应 CSV 列 | Y 轴标签 |
|--------|------|------------|----------|
| ① | 主轴振动 | 第 4 列 **Z 轴**（主轴轴向振动） | 振动幅值 |
| ② | X 轴振动 | 第 2 列 | 振动幅值 |
| ③ | Y 轴振动 | 第 3 列 | 振动幅值 |
| ④ | 三向力合力 | 第 5 列 | 力合力 (N) |

- X 轴统一为**时间（秒）**，四行共享 X 轴，便于纵向对齐各通道的相位/幅值关系。
- 主轴通道（①）以强调色标注，突出关键轴向。
- 信号点数超过 20000 时自动降采样，控制绘制开销与文件体积。
- 绘图失败**不会**阻断诊断主流程（仅报告内缺失图路径提示）。
- 实时监控模式（`monitor`）同样在生成报告前绘图，便于人工核对基线偏离段。

## 实时监控模式（z-score 基线偏离）

复用离线建立的**稳态基线**（`script/models/baseline_stats.json`），对在线信号每段 50 维特征计算 z-score 偏离，实现加工过程实时看护（🟢正常 / 🟡关注 / 🔴报警）。

- **基线来源**：由 `script/train_fusion_model.py` 产出（各特征在稳定工况下的均值 μ 与标准差 σ）。v3.0 主线 `train_all.py` 不生成该文件，启用监控前需单独运行 `python -m script.train_fusion_model`。
- **公式**：`z_i = |x_i − μ_i| / σ_i`；段评分取 50 维中最大 z。
- **阈值**：段最大 z `< 2.0` 🟢正常 / `2.0~3.5` 🟡关注 / `≥ 3.5` 🔴报警。
- **趋势**：前半段均 z vs 后半段均 z，后者 > 前者×1.5 为 📈劣化，< 前者×0.5 为 📉好转，其余 ↔️稳定。
- 监控入口 `monitor_csv(csv_path)` **已实现**：命令 `python -m script.chatter_diagnosis_skill monitor signal.csv` 即可运行（需先 `python -m script.train_fusion_model` 建立基线）。报告样例见下「监控报告样例」。

## 50 维特征定义

每个振动轴使用**独立的 15 特征集**（顺序不可改变，与训练模型一一对应）：

**X 轴（15 维）**
```
Time Clearance Factor, Frequency Power Spectrum Clearance, Time Peak,
Time Peak-to-Peak, Time RMS, Time Std, STFT Mean, Time Variance,
Time Signal Energy, Frequency Spectral Energy, STFT Total Energy,
Time-Frequency Entropy, Frequency Power Spectrum Peak,
Frequency Variance, Time Shape Factor
```

**Y 轴（15 维）**
```
Time Peak-to-Peak, STFT Mean, Time Std, Time RMS, Time Peak,
Frequency Power Spectrum Clearance, Time Clearance Factor,
Frequency Mean Square Frequency, Frequency Spectral Centroid,
Time-Frequency Entropy, Frequency Variance, Time Skewness, Time Mean,
STFT Total Energy, Time Signal Energy
```

**Z 轴（15 维）**
```
Time Clearance Factor, Time Peak-to-Peak,
Frequency Power Spectrum Clearance, Time Peak, Time Shape Factor,
STFT Mean, Time Std, Time RMS, Time Impulse Factor, Time Crest Factor,
Time Kurtosis, Frequency Variance, Frequency Mean Square Frequency,
Frequency Peak Count, Frequency Spectral Centroid
```

**三向力合力（5 维）**
```
force_freq_variance, force_peak2peak, force_impulse_factor,
force_peak, force_crest_factor
```

## 快速开始

### 1. 安装依赖

```bash
cd script
pip install -r requirements.txt
```

### 2. 训练（首次使用或数据更新后）

```bash
# 在 skill 根目录执行
python -m script.train_all [训练数据.csv]
```

训练数据 CSV 格式（6 列）：`时间, X振动, Y振动, Z振动, 三向力合力, 工况标签`。
不传参数时默认读取 `C:\Users\siat\Desktop\平衡后_三种工况数据.csv`。

> 如需启用**实时监控模式**，另需构建稳态基线（仅需一次，数据更新后重建）：
> ```bash
> python -m script.train_fusion_model
> ```
> 该脚本产出 `script/models/baseline_stats.json`（监控模式专用，不参与上述诊断管线）。

训练产物（保存到 `script/models/`）：

| 文件 | 说明 |
|------|------|
| `sae_model.pth` | SAE 稀疏自编码器（50→16） |
| `scaler.pkl` | 50 维特征标准化器 |
| `scaler16.pkl` | 16 维编码特征标准化器 |
| `fusion_model.pkl` | 弱分类器 + F1 权重 + Stacking |
| `baseline_stats.json` | 实时监控基线（μ/σ），由 `train_fusion_model.py` 产出；仅监控模式需要 |

### 3. 诊断

```bash
# 命令行
python -m script.chatter_diagnosis_skill signal.csv

# 或 Python 调用
python -c "
from script.chatter_diagnosis_skill import diagnose_csv
print(diagnose_csv('signal.csv'))
"
```

## CSV 输入格式（诊断用）

| 列序 | 内容 | 说明 |
|------|------|------|
| 1 | 时间 | 秒 |
| 2 | X 轴振动 | 振动幅值 |
| 3 | Y 轴振动 | 振动幅值 |
| 4 | Z 轴振动（即主轴振动） | 振动幅值 |
| 5 | 三向力合力 | N |

**采样点要求：≥ 256 点**（不足一个分段无法诊断）。列按位置读取，前 5 列顺序必须一致。

## 输出示例

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  颤振诊断报告 (SAE + Stacking)
  生成时间: 2026-07-30 15:30:15
  信号文件: signal.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

信号概况:
  时长: 2.05s
  采样点数: 2048
  采样率: 1000.0Hz
  分段数: 8 (256点/段)
  信号图: signal_signal.png
         (4×1 时序图：①主轴/Z振动 ②X振动 ③Y振动 ④三向力合力，X轴=时间)

逐段诊断 (弱分类器投票 | Stacking最终):
  段1: 0.00s-0.26s  投票=稳定 | Stacking=稳定 (置信度 96.3%)
  段2: 0.26s-0.51s  投票=轻微颤振 | Stacking=轻微颤振 (置信度 88.1%)
  ...

整体结论: 轻微颤振
状态分布: 稳定×3, 轻微颤振×5

工艺建议: 检测到轻微颤振 — 建议适当降低进给速度并持续观察。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 监控报告样例（`monitor` 模式）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  颤振实时监控报告 (z-score 基线偏离)
  生成时间: 2026-07-30 16:10:02
  信号文件: signal.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

信号概况:
  时长: 2.05s
  采样点数: 2048
  采样率: 1000.0Hz
  分段数: 8 (256点/段)

逐段监控 (最大 z | 评级):
  段1: 0.00s-0.26s  z=1.21 🟢正常
  段2: 0.26s-0.51s  z=2.43 🟡关注
  段3: 0.51s-0.77s  z=3.82 🔴报警
  ...

整体评估:
  最大 z: 3.82   平均 z: 2.15
  状态: 🔴报警  (🟢4 / 🟡3 / 🔴1)
  趋势: 📈劣化  (前半均z=1.40, 后半均z=2.90)

建议: 检测到报警段 — 立即检查加工状态，降低进给速度、减小切深、避开共振区。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> 监控模式无需 SAE / scaler16，仅依赖 `baseline_stats.json` 与 50 维特征，因此可与诊断模式并行使用。

## 文件结构

```
skill/
├── SKILL.md                          # 本文件 — 技能定义与使用说明
├── script/
│   ├── __init__.py
│   ├── chatter_diagnosis_skill.py    # 诊断入口（诊断 + 实时监控双模式）
│   ├── feature_extractor.py          # 50维特征提取（X/Y/Z独立特征集+力）
│   ├── signal_plot.py                # 4×1 原始信号时序图（v3.1，不依赖模型）
│   ├── train_sae_model.py            # SAE 定义与训练（对齐2.2参考代码）
│   ├── train_all.py                  # 一键诊断训练（SAE+弱分类器+Stacking，v3.0 主线）
│   ├── train_fusion_model.py         # 构建监控基线（产出 baseline_stats.json）
│   ├── requirements.txt
│   └── models/                       # 训练产物
│       ├── sae_model.pth
│       ├── scaler.pkl
│       ├── scaler16.pkl
│       └── fusion_model.pkl
├── reference/                        # 参考文档
├── CLAUDE.md
└── README.md
```

## 关键一致性约束（修改代码前必读）

1. **分段长度**：训练与推理均为 **256 点/段**；训练时 50% 重叠增广，推理时无重叠。改动任一侧必须同步另一侧并重新训练。
2. **特征顺序**：X/Y/Z 三轴特征集互不相同且顺序固定，改动 `feature_extractor.py` 后必须重新训练全部模型。
3. **两级归一化**：50 维用 `scaler.pkl`，SAE 编码后的 16 维用 `scaler16.pkl`，推理时缺一不可。
4. **SAE 架构**：编码器 50→512→256→128→16（BatchNorm+ReLU+Dropout），与 checkpoint 权重严格绑定，不可改层数。
5. **模型文件键名**：`fusion_model.pkl` 必须包含 `classifiers` / `weights` / `stacking` 三个键。
6. **信号图（v3.1）**：绘图逻辑集中于 `script/signal_plot.py`，四子图顺序固定为
   `[Z(主轴), X, Y, 力]`，与 CSV 列序为 `(时间, X, Y, Z, 力)` 的映射关系见上表。
   改动子图顺序或通道映射时，须同步 `_SIGNAL_LAYERS` 与本文档「信号时序图」小节。

## 前置条件与错误处理

| 场景 | 错误信息 | 处理方式 |
|------|---------|---------|
| 信号 < 256 点 | "信号过短" | 增加采样时间 |
| 模型文件缺失 | "SAE 模型不存在" | 运行 `python -m script.train_all` |
| scaler16 缺失 | "Scaler16 不存在" | 旧版模型，需重新训练 |
| CSV 格式错误 | "CSV列数不足" | 检查列数（≥5列）和编码 |
