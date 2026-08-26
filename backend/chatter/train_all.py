"""
统一训练脚本 - 一键训练 SAE + 融合分类器

运行:
    python train_all.py                          # 使用默认路径
    python train_all.py --data /path/to/data     # 指定训练数据路径
    python train_all.py --data ./training_data   # 相对路径
    set CHATTER_DATA_DIR=D:\\data  && python train_all.py   # 环境变量

输出: models/ 目录下的 sae_model.pth, scaler.pkl, fusion_model.pkl
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import (
    extract_vibration_features,
    extract_force_features,
    VIB_FEATURE_NAMES,
    FORCE_FEATURE_NAMES,
)

# 默认训练数据路径
DEFAULT_DATA_DIR = r"D:\senordata\virbrant_clear"


def get_data_dir():
    """获取训练数据目录: 命令行参数 > 环境变量 > 默认值"""
    # 环境变量
    env_dir = os.environ.get('CHATTER_DATA_DIR', '')
    if env_dir:
        return env_dir
    return DEFAULT_DATA_DIR


def extract_features_from_excel(excel_path, n_features=26):
    """
    从现有特征 Excel 文件加载数据。
    格式: 第1列为标签(Condition), 其后各列为特征。
    """
    if not os.path.exists(excel_path):
        return None, None, None

    df = pd.read_excel(excel_path)
    if df.shape[1] < n_features + 1:
        print(f"  ⚠️ 文件列数不足: {df.shape[1]}, 期望至少 {n_features + 1}")
        return None, None, None

    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    feature_names = [str(c).strip() for c in df.columns[1:]]

    return features, labels, feature_names


# 运行时特征名 (feature_extractor.VIB_FEATURE_NAMES) -> 特征Excel中的列名
VIB_EXCEL_NAME_MAP = {
    'Clearance_Factor': 'Time Clearance Factor',
    'Power_Spectrum_Clearance': 'Frequency Power Spectrum Clearance',
    'Peak': 'Time Peak',
    'Peak_to_Peak': 'Time Peak-to-Peak',
    'RMS': 'Time RMS',
    'Std': 'Time Std',
    'STFT_Mean': 'STFT Mean',
    'Variance': 'Time Variance',
    'Signal_Energy': 'Time Signal Energy',
    'Spectral_Energy': 'Frequency Spectral Energy',
    'STFT_Total_Energy': 'STFT Total Energy',
    'Time_Frequency_Entropy': 'Time-Frequency Entropy',
    'Power_Spectrum_Peak': 'Frequency Power Spectrum Peak',
    'Frequency_Variance': 'Frequency Variance',
    'Shape_Factor': 'Time Shape Factor',
}


def select_fixed_features(all_features, sensor_type='vib', feature_names=None):
    """
    按列名选择固定的15个(vibration)或5个(force)特征。

    - vib: 按 VIB_EXCEL_NAME_MAP 将运行时15个特征名映射到Excel列名后取列
    - force: 取标签列之后的前5列
    """
    if sensor_type == 'vib':
        if not feature_names:
            print("  ⚠️ 缺少特征列名，无法按名选择振动特征")
            return None
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        missing = [VIB_EXCEL_NAME_MAP[n] for n in VIB_FEATURE_NAMES
                   if VIB_EXCEL_NAME_MAP[n] not in name_to_idx]
        if missing:
            print(f"  ⚠️ Excel 缺少特征列: {missing}")
            return None
        indices = [name_to_idx[VIB_EXCEL_NAME_MAP[n]] for n in VIB_FEATURE_NAMES]
        return all_features[:, indices]

    elif sensor_type == 'force' and all_features.shape[1] >= 5:
        # 力传感器取前5个特征
        return all_features[:, :5]

    return all_features


def prepare_training_data(data_dir=None):
    """
    准备训练数据: 从现有特征Excel文件加载并选择固定50维特征。

    Args:
        data_dir: 训练数据根目录路径
    """
    data_dir = data_dir or get_data_dir()

    print("=" * 60)
    print("准备训练数据...")
    print(f"数据路径: {data_dir}")
    print("=" * 60)

    # 尝试从现有特征Excel文件加载
    vib_paths = {
        'Z': os.path.join(data_dir, '提取的特征', 'Z', '1-5', '振动数据_特征提取结果.xlsx'),
        'X': os.path.join(data_dir, '提取的特征', 'X', '1-5', '振动数据_特征提取结果.xlsx'),
        'Y': os.path.join(data_dir, '提取的特征', 'Y', '1-5', '振动数据_特征提取结果.xlsx'),
    }
    force_path = os.path.join(data_dir, '提取的特征', '力', '振动数据_特征提取结果.xlsx')

    all_data = []
    labels = None
    total_samples = 4270

    # 加载振动传感器数据
    for name, path in vib_paths.items():
        print(f"\n加载{name}轴振动数据: {path}")
        features, lbls, feat_names = extract_features_from_excel(path)
        if features is None:
            print(f"  ❌ 文件不存在或格式错误，请确保已运行 1.3_putFeatures.py")
            return None
        features = features[:total_samples]
        if labels is None:
            labels = lbls[:total_samples]
        selected = select_fixed_features(features, sensor_type='vib', feature_names=feat_names)
        if selected is None:
            print(f"  ❌ 特征列名不匹配，无法选择15个固定特征")
            return None
        all_data.append(selected)
        print(f"  ✅ 形状: {selected.shape} (选择15个固定特征)")

    # 加载力传感器数据
    print(f"\n加载力传感器数据: {force_path}")
    force_features, _, _ = extract_features_from_excel(force_path, n_features=5)
    if force_features is None:
        print(f"  ❌ 力传感器文件不存在")
        return None
    force_features = force_features[:total_samples]
    force_selected = select_fixed_features(force_features, sensor_type='force')
    all_data.append(force_selected)
    print(f"  ✅ 形状: {force_selected.shape} (选择5个固定特征)")

    # 行数对齐检查
    row_counts = [d.shape[0] for d in all_data]
    if len(set(row_counts)) > 1:
        print(f"\n❌ 各传感器样本行数不一致: 振动X/Y/Z + 力 = {row_counts}")
        print("   四个特征文件必须逐行对齐(同一信号切片)，请检查力传感器特征文件")
        return None

    # 合并
    X = np.hstack(all_data)
    print(f"\n合并后特征维度: {X.shape} (期望 4270×50)")
    print(f"标签分布: {np.bincount(labels.astype(int))}")

    return X, labels


def main(data_dir=None):
    print("颤振诊断 Skill - 统一训练脚本")
    print("=" * 60)

    # 1. 准备数据
    data = prepare_training_data(data_dir)
    if data is None:
        print("\n❌ 数据准备失败，请检查训练数据路径")
        print(f"   当前路径: {data_dir or get_data_dir()}")
        print(f"   可通过以下方式指定:")
        print(f"     python train_all.py --data /your/data/path")
        print(f"     或设置环境变量: set CHATTER_DATA_DIR=D:\\your\\data\\path")
        return

    X, labels = data

    # 2. 训练 SAE
    print("\n" + "=" * 60)
    print("训练 SAE 模型...")
    print("=" * 60)
    from train_sae_model import train_sae, save_model, visualize_training
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练/验证
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, labels, test_size=0.1, random_state=42, stratify=labels
    )

    model, history = train_sae(X_train, X_val, y_train, y_val)

    # 保存 SAE
    import torch
    import json
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    with torch.no_grad():
        X_encoded = model.get_encoded(torch.FloatTensor(X_scaled)).numpy()

    # 计算信息保留率
    with torch.no_grad():
        X_recon = model(torch.FloatTensor(X_scaled))[1].numpy()
    mse = np.mean((X_scaled - X_recon) ** 2)
    info_ret = (1 - mse / np.var(X_scaled)) * 100

    # 评估
    X_enc_train, X_enc_val = X_encoded[:len(X_train)], X_encoded[len(X_train):]
    svm = SVC(kernel='rbf'); svm.fit(X_enc_train, y_train)
    svm_acc = accuracy_score(y_val, svm.predict(X_enc_val))
    rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_enc_train, y_train)
    rf_acc = accuracy_score(y_val, rf.predict(X_enc_val))

    # 保存 50 维原始特征空间的基线统计（供规则模式分析使用）
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    CLASS_NAMES_BASELINE = ['稳定', '轻微颤振', '严重颤振']
    vib_names = [f"Vib{i}_{n}" for i in range(3) for n in VIB_FEATURE_NAMES]
    force_names = [f"Force_{n}" for n in FORCE_FEATURE_NAMES]
    all_feat_names = vib_names + force_names  # 50 个特征名
    baseline_50d = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES_BASELINE):
        mask = labels.astype(int) == cls_idx
        if not mask.any():
            continue
        for feat_idx, feat_name in enumerate(all_feat_names):
            key = f"{cls_name}_{feat_name}"
            vals = X[mask, feat_idx]
            baseline_50d[key] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()) if vals.std() > 1e-12 else 1e-6,
            }
    baseline_path = os.path.join(MODEL_DIR, 'baseline_stats.json')
    # 合并原有的 SAE 空间基线与新的 50D 空间基线
    existing_baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding='utf-8') as f:
            existing_baseline = json.load(f)
    existing_baseline.update(baseline_50d)
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(existing_baseline, f, indent=2)
    print(f"基线统计已更新 (含 50D 特征空间): {baseline_path}")

    save_model(model, scaler, info_ret, svm_acc, rf_acc, history)
    visualize_training(history)

    # 3. 保存编码特征供融合训练
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    enc_df = pd.DataFrame(X_encoded, columns=[f'SAE_{i+1}' for i in range(16)])
    enc_df['label'] = labels
    enc_df.to_csv(os.path.join(MODEL_DIR, 'sae_encoded_features.csv'), index=False)
    print(f"\n编码特征已保存: {MODEL_DIR}/sae_encoded_features.csv")

    # 4. 训练融合模型
    print("\n" + "=" * 60)
    print("训练融合分类器...")
    print("=" * 60)
    from train_fusion_model import train_fusion
    train_fusion()

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"模型文件已保存至: {MODEL_DIR}")
    print(f"  - sae_model.pth (SAE模型)")
    print(f"  - scaler.pkl (标准化器)")
    print(f"  - fusion_model.pkl (融合分类器)")
    print(f"  - baseline_stats.json (基线统计)")
    print("\n现在可以使用 chatter_diagnosis_skill.py 进行诊断:")
    print(f"  python chatter_diagnosis_skill.py <your_signal.csv>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='颤振诊断 Skill - 统一训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python train_all.py                          # 使用默认路径
  python train_all.py --data /path/to/data     # 指定训练数据路径
  python train_all.py --data ./training_data   # 相对路径

环境变量:
  CHATTER_DATA_DIR    训练数据根目录 (优先级低于 --data)
'''
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='训练数据根目录路径 (默认: %s)' % DEFAULT_DATA_DIR,
    )
    args = parser.parse_args()
    main(data_dir=args.data)
