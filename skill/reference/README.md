# 颤振诊断 Skill — 技术参考手册（v3.0.0）

> 配套权威文档：`SKILL.md`（v3.0.0）。本文档描述实现细节，凡与 `SKILL.md` 冲突以 `SKILL.md` 为准。
> 本 Skill 提供两种模式：**颤振诊断模式**（SAE + Stacking 三分类）与 **实时监控模式**（z-score 基线偏离）。两者共用 50 维特征与同一份传感器 CSV。

## 1. 特征提取详解（50 维）

每个振动轴使用**独立的 15 特征集**（顺序固定，与训练模型一一对应），力信号 5 维，共 50 维。三轴顺序为 **X / Y / Z**，并非“主轴振动”。

### X 轴（15 维，索引 0–14）
| 索引 | 特征名 (代码) | 域 |
|------|---------------|-----|
| 0 | X_Clearance_Factor (Time Clearance Factor) | 时域 |
| 1 | X_Power_Spectrum_Clearance | 频域 |
| 2 | X_Peak (Time Peak) | 时域 |
| 3 | X_Peak_to_Peak | 时域 |
| 4 | X_RMS | 时域 |
| 5 | X_Std | 时域 |
| 6 | X_STFT_Mean | 时频 |
| 7 | X_Variance | 时域 |
| 8 | X_Signal_Energy | 时域 |
| 9 | X_Spectral_Energy | 频域 |
| 10 | X_STFT_Total_Energy | 时频 |
| 11 | X_Time_Frequency_Entropy | 时频 |
| 12 | X_Power_Spectrum_Peak | 频域 |
| 13 | X_Frequency_Variance | 频域 |
| 14 | X_Shape_Factor | 时域 |

### Y 轴（15 维，索引 15–29）
| 索引 | 特征名 (代码) | 域 |
|------|---------------|-----|
| 15 | Y_Peak_to_Peak | 时域 |
| 16 | Y_STFT_Mean | 时频 |
| 17 | Y_Std | 时域 |
| 18 | Y_RMS | 时域 |
| 19 | Y_Peak | 时域 |
| 20 | Y_Power_Spectrum_Clearance | 频域 |
| 21 | Y_Clearance_Factor | 时域 |
| 22 | Y_Mean_Square_Frequency | 频域 |
| 23 | Y_Spectral_Centroid | 频域 |
| 24 | Y_Time_Frequency_Entropy | 时频 |
| 25 | Y_Frequency_Variance | 频域 |
| 26 | Y_Skewness | 时域 |
| 27 | Y_Mean | 时域 |
| 28 | Y_STFT_Total_Energy | 时频 |
| 29 | Y_Signal_Energy | 时域 |

### Z 轴（15 维，索引 30–44）
| 索引 | 特征名 (代码) | 域 |
|------|---------------|-----|
| 30 | Z_Clearance_Factor | 时域 |
| 31 | Z_Peak_to_Peak | 时域 |
| 32 | Z_Power_Spectrum_Clearance | 频域 |
| 33 | Z_Peak | 时域 |
| 34 | Z_Shape_Factor | 时域 |
| 35 | Z_STFT_Mean | 时频 |
| 36 | Z_Std | 时域 |
| 37 | Z_RMS | 时域 |
| 38 | Z_Impulse_Factor | 时域 |
| 39 | Z_Crest_Factor | 时域 |
| 40 | Z_Kurtosis | 时域 |
| 41 | Z_Frequency_Variance | 频域 |
| 42 | Z_Mean_Square_Frequency | 频域 |
| 43 | Z_Peak_Count | 频域 |
| 44 | Z_Spectral_Centroid | 频域 |

### 力信号（5 维，索引 45–49）
| 索引 | 特征名 (代码) |
|------|---------------|
| 45 | Force_Freq_Variance |
| 46 | Force_Peak2Peak |
| 47 | Force_Impulse_Factor |
| 48 | Force_Peak |
| 49 | Force_Crest_Factor |

## 2. 模型架构（颤振诊断模式）

### 稀疏自编码器 (SAE)
```
输入: 50 维
  ↓ 编码器 (Linear + BatchNorm1d + ReLU + Dropout)
  512 → 256 → 128 → 16
  ↓ 解码器 (Linear + BatchNorm1d + ReLU + Dropout)
  128 → 256 → 512 → 50
损失: MSE(重构) + sparsity_weight × KL(稀疏, 目标 0.05)
```
- 编码器结构 `50→512→256→128→16`，与 checkpoint 权重严格绑定，**不可改层数**。
- 稀疏正则采用 **KL 散度**（非 L1）。

### 融合分类器 (Stacking)
```
第一层（6 个弱分类器）:
  LightGBM / CatBoost(可选) / SVM(RBF) / ExtraTrees / KNN(k=5) / MLP
  └─ F1 自适应加权投票（参考诊断）
第二层: Logistic Regression（元分类器, cv=5, predict_proba, passthrough）
```
- 弱分类器权重由验证集 F1 自适应归一化得到。

## 3. 实时监控模式（z-score 基线偏离）

以离线阶段建立的**稳态基线**为参考，对在线信号的每段 50 维特征计算 z-score 偏离，实现加工过程的实时看护（🟢正常 / 🟡关注 / 🔴报警）。

### 基线来源
- 由 `script/train_fusion_model.py` 训练时产出 `script/models/baseline_stats.json`（各特征在稳定工况下的均值 μ 与标准差 σ）。
- 该脚本同时产出 `fusion_model.pkl`；若需启用监控模式，请确保其已运行（v3.0 主线 `train_all.py` 不生成基线文件）。

### z-score 计算公式
```
对特征 i:
  z_i = |x_i - μ_i| / σ_i
  x_i = 当前段提取的特征值
  μ_i = 基线中该特征的均值
  σ_i = 基线中该特征的标准差
```

### 段评分
```
段最大 z = max(z_0, z_1, ..., z_49)
```

### 告警阈值
| 段最大 z | 状态 | 颜色 |
|----------|------|------|
| < 2.0 | 正常 | 🟢 |
| 2.0 ~ 3.5 | 关注 | 🟡 |
| ≥ 3.5 | 报警 | 🔴 |

### 趋势分析
- 计算前半段 z 均值 vs 后半段 z 均值
- 后半段 > 前半段 × 1.5 → 📈 劣化趋势
- 后半段 < 前半段 × 0.5 → 📉 好转趋势
- 其他 → ↔️ 稳定

## 4. 信号处理参数
| 参数 | 值 | 说明 |
|------|-----|------|
| 最小采样点 | 256 | 至少一个完整段 |
| 分段大小 | 256 点 | 训练 50% 重叠增广，推理无重叠 |
| 采样率 | 自动探测 | 由时间列差分中位数计算 |
| CSV 编码 | UTF-8 / GBK / GB18030 / Latin-1 自动探测 | 兼容国产采集系统 |
| 列读取 | 前 5 列按位置读取 | 时间, X振动, Y振动, Z振动, 三向力合力 |

## 5. 训练流程
```bash
# 1. 准备训练数据（单个 CSV，6 列，无表头）:
#    时间, X振动, Y振动, Z振动, 三向力合力, 工况标签
#    默认读取 C:\Users\siat\Desktop\平衡后_三种工况数据.csv

# 2. 一键训练（v3.0 诊断主线；位置参数可选，无 --data 等开关）
python -m script.train_all [训练数据.csv]

# 3. 构建监控基线（启用实时监控模式时额外运行，产出 baseline_stats.json）
python -m script.train_fusion_model

# 4. 输出模型（script/models/）
#    诊断模式: sae_model.pth / scaler.pkl / scaler16.pkl / fusion_model.pkl
#    监控模式: baseline_stats.json（来自 train_fusion_model）
```
> 说明：诊断训练参数 `epochs=300`、`lr=0.001`、`encoding_dim=16` 在 `train_all.py` 中硬编码，非命令行可配。
> `script/train_fusion_model.py` 为旧版独立脚本，输出 `fusion_model.pkl` 与 `baseline_stats.json` / `sae_encoded_features.csv`，**仅用于为实时监控模式构建基线**，不参与诊断管线。

## 6. 诊断调用
```bash
# 命令行
python -m script.chatter_diagnosis_skill signal.csv

# Python API
from script.chatter_diagnosis_skill import diagnose_csv
print(diagnose_csv('signal.csv'))
```
- 输出：逐段 `弱分类器投票 | Stacking 最终` + 整体结论（稳定 / 轻微颤振 / 严重颤振）。
- 本技能**不含** FastAPI 后端与 `backend.chatter` 模块（旧文档中的相关描述已废弃）。
- 实时监控模式的告警逻辑基于上述 z-score 公式与基线文件，可对接在线采集流做逐段评分。

## 7. 版本说明（避免与旧文档混淆）
| 版本 | 模式 | 关键差异 |
|------|------|---------|
| v1.0.0（`reference/chatter_diagnosis.md`，已废弃） | 多层融合 | 主轴振动列、≥512 点 |
| v2.0.0（`skills.json` 旧描述，已废弃） | 5 种模式 + 实时监控 | 多模式命名，监控实现位于后端 |
| **v3.0.0（现行）** | **诊断模式(SAE+Stacking) + 实时监控(z-score)** | X/Y/Z 三轴、≥256 点、两级归一化（scaler + scaler16），监控复用 baseline_stats.json |
