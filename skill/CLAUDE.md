# 颤振诊断 Skill（v3.1.0）

基于 SAE 稀疏自编码器 + Stacking 集成的加工颤振自动诊断系统，并提供**实时 z-score 基线监控**模式。

> 权威文档：`SKILL.md`（v3.0.0）。本文档仅作快速索引，凡与 `SKILL.md` 冲突之处一律以 `SKILL.md` 为准。

## 目录结构

```
skill/
├── SKILL.md                    # 技能定义（元数据 + 完整使用说明，权威）
├── skills.json                 # 技能注册表（version 3.0.0）
├── README.md                   # 快速入门
├── CLAUDE.md                   # 本文件
├── script/                     # 可执行脚本
│   ├── chatter_diagnosis_skill.py    # 诊断主入口（diagnose_csv）
│   ├── feature_extractor.py          # 50维特征提取（X/Y/Z 独立特征集 + 力）
│   ├── signal_plot.py                # 4×1 原始信号时序图（v3.1，不依赖模型）
│   ├── train_sae_model.py            # SAE 定义与训练（50→512→256→128→16）
│   ├── train_fusion_model.py         # 融合训练 + 产出监控基线 baseline_stats.json
│   ├── train_all.py                  # 一键诊断训练（SAE+弱分类器+Stacking，v3.0 主线）
│   ├── models/                       # 训练产物
│   └── requirements.txt              # 依赖包
└── reference/                   # 参考文档
    ├── README.md                     # 技术参考手册（v3.0）
    └── chatter_diagnosis.md          # 旧版(v1.0.0)技能定义（历史归档，已废弃）
```

## 快速使用

```bash
# 安装依赖
cd script && pip install -r requirements.txt

# 诊断训练（首次 / 数据更新后；可选位置参数指定训练 CSV，默认读取桌面数据）
python -m script.train_all [训练数据.csv]

# 构建监控基线（启用实时监控模式时额外运行，产出 baseline_stats.json）
python -m script.train_fusion_model

# 运行诊断
python -m script.chatter_diagnosis_skill signal.csv

# Python API
python -c "from script.chatter_diagnosis_skill import diagnose_csv; print(diagnose_csv('signal.csv'))"
```

## 关键事实（与代码严格一致）

- 两种模式：
  - **颤振诊断模式**：SAE + Stacking 三分类（🟢空载/🟢稳定/🔴颤振）。
  - **实时监控模式**：基于稳态基线的 z-score 偏离监测（🟢<2.0 / 🟡2.0~3.5 / 🔴≥3.5），复用 `baseline_stats.json`。
- CSV 诊断格式：前 5 列按位置读取 —— `时间, X振动, Y振动, Z振动, 三向力合力`。其中 Z 轴即主轴轴向振动；v3.1 信号图中第①行"主轴振动"即对应 Z 列。
- 训练数据格式：6 列 —— `时间, X振动, Y振动, Z振动, 三向力合力, 工况标签`。
- 采样点要求：**≥ 256 点**；分段 **256 点/段**（训练 50% 重叠增广，推理无重叠）。
- 特征：50 维（X/Y/Z 各 15 + 力 5），三轴特征集互不相同且顺序固定。
- 降维：SAE `50 → 512 → 256 → 128 → 16`（BatchNorm+ReLU+Dropout，KL 稀疏正则）。
- 分类：6 个弱分类器（LightGBM / CatBoost / SVM / ExtraTrees / KNN / MLP）+ Stacking（LR 元学习器, cv=5）。
- 必需模型文件（置于 `script/models/`）：
  - 诊断模式：`sae_model.pth`、`scaler.pkl`、`scaler16.pkl`、`fusion_model.pkl`
  - 监控模式：`baseline_stats.json`（由 `train_fusion_model.py` 产出）

## 历史版本提示（避免与旧文档混淆）
- **v1.0.0**（`reference/chatter_diagnosis.md`）：多层融合 + 主轴振动列 + 512 点 —— 已废弃（监控模式在现行版中保留）。
- **v2.0.0**（`skills.json` 旧描述）：5 种模式 + 实时监控 —— 已废弃（现行合并为“诊断 + 实时监控”双模式）。
- **v3.0.0**（现行，`SKILL.md`）：诊断模式 SAE + Stacking，**保留实时 z-score 监控模式**。
- **v3.1.0**（现行补强，`SKILL.md`）：新增 4×1 原始信号时序图（v3.1），读入即绘图、不依赖模型，输出 `<csv_stem>_signal.png`。
