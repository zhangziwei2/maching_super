"""
颤振诊断 — 统一训练脚本 v3.0
方法严格对齐参考代码：
  - 降维:   2.2_SAEDimensionreduction.py (StandardScaler → SAE 50→512→256→128→16)
  - 分类:   3.1_Decision_Integration.py  (16维再标准化 → 弱分类器 → F1加权 → Stacking)

流程：
  1. 读取原始 CSV → 分段(256点, 50%重叠) → 每段提取50维特征
  2. 分层抽样划分训练/验证集
  3. StandardScaler 归一化 50 维特征
  4. SAE 降维 50→16 (2.2 方法)
  5. 16 维编码特征再次 StandardScaler (3.1 方法)
  6. 弱分类器训练: LightGBM / CatBoost / SVM / ExtraTrees / KNN / MLP
  7. F1 自适应加权 (3.1 adaptive_weight_optimization)
  8. Stacking 集成 (LR 元学习器, cv=5, predict_proba, passthrough)
  9. 保存模型: sae_model.pth, scaler.pkl, scaler16.pkl, fusion_model.pkl

用法：
  python -m script.train_all [csv_path]
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from script.feature_extractor import extract_50_features, ALL_FEATURE_NAMES
from script.train_sae_model import SparseAutoencoder, train_sae, save_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import torch
import joblib

warnings.filterwarnings('ignore')

# ==================== 路径与常量 ====================
SKILL_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = SKILL_DIR / 'script' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = r"C:\Users\siat\Desktop\平衡后_三种工况数据.csv"

SEGMENT_SIZE = 256        # 每段点数（训练与推理必须一致）
OVERLAP_RATIO = 0.5       # 训练时 50% 重叠做数据增强
ENCODING_DIM = 16         # SAE 编码维度


# ==================== 数据加载与分段 ====================

def read_csv(csv_path: str) -> dict:
    """读取原始 CSV: 时间, X振动, Y振动, Z振动, 三向力合力, 工况标签"""
    df_raw = pd.read_csv(csv_path, encoding='utf-8', header=None)
    df = df_raw.iloc[1:].copy().reset_index(drop=True)  # 跳过表头行
    for i in range(5):
        df[i] = pd.to_numeric(df[i], errors='coerce')

    time = df[0].values
    x, y, z, force = df[1].values, df[2].values, df[3].values, df[4].values
    labels_raw = df[5].values if df.shape[1] >= 6 else None
    labels = np.array([str(v).strip() for v in labels_raw]) if labels_raw is not None else None

    # 标签映射: 稳定→0, 轻微/空载→1, 颤振/严重→2
    LABEL_MAP = {}
    if labels is not None:
        for k in sorted(set(labels)):
            if '稳定' in k or '正常' in k:
                LABEL_MAP[k] = 0
            elif '空载' in k or '轻微' in k or '轻度' in k:
                LABEL_MAP[k] = 1
            elif '颤振' in k or '严重' in k or '重度' in k:
                LABEL_MAP[k] = 2

    valid = ~(np.isnan(time) | np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(force))
    if labels is not None:
        labels_num = np.array([LABEL_MAP.get(s, -1) for s in labels])
        valid &= (labels_num >= 0)
        labels = labels_num[valid].astype(int)

    dt = np.median(np.diff(time[valid]))
    fs = 1.0 / dt if dt > 0 else 1000.0

    return {
        'time': time[valid], 'x': x[valid], 'y': y[valid], 'z': z[valid],
        'force': force[valid], 'labels': labels,
        'sampling_rate': fs, 'n_samples': int(np.sum(valid)),
    }


def segment_data(data: dict) -> tuple:
    """分段（256点, 50%重叠），标签取段内中位数"""
    n = len(data['time'])
    stride = max(1, int(SEGMENT_SIZE * (1 - OVERLAP_RATIO)))
    segments, seg_labels = [], []

    for start in range(0, n - SEGMENT_SIZE + 1, stride):
        end = start + SEGMENT_SIZE
        segments.append({
            'x': data['x'][start:end], 'y': data['y'][start:end],
            'z': data['z'][start:end], 'force': data['force'][start:end],
        })
        if data['labels'] is not None:
            seg_labels.append(int(np.median(data['labels'][start:end])))

    return segments, (np.array(seg_labels, dtype=int) if seg_labels else None)


def extract_all_features(segments: list, fs: float) -> np.ndarray:
    return np.array([
        extract_50_features(s['x'], s['y'], s['z'], s['force'], sampling_rate=fs)
        for s in segments
    ])


# ==================== 3.1 方法: 弱分类器 + Stacking ====================

def train_base_classifiers(X_train, y_train, X_val, y_val):
    """弱分类器训练 — 对齐 3.1 train_base_classifiers_with_optimization"""
    classifiers, performances = {}, {}

    # 1. LightGBM
    print("  训练 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.0, min_child_samples=20,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb_model.fit(X_train, y_train)
    classifiers['lightgbm'] = lgb_model

    # 2. CatBoost（可选）
    try:
        from catboost import CatBoostClassifier
        print("  训练 CatBoost...")
        cat = CatBoostClassifier(iterations=800, depth=8, learning_rate=0.05,
                                 verbose=False, random_seed=42)
        cat.fit(X_train, y_train)
        classifiers['catboost'] = cat
    except ImportError:
        print("  CatBoost 未安装，跳过")

    # 3. SVM
    print("  训练 SVM...")
    svm = SVC(C=50, kernel='rbf', gamma='scale', probability=True,
              cache_size=2000, random_state=42)
    svm.fit(X_train, y_train)
    classifiers['svm'] = svm

    # 4. ExtraTrees
    print("  训练 ExtraTrees...")
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', bootstrap=False,
        random_state=42, n_jobs=-1,
    )
    et.fit(X_train, y_train)
    classifiers['extratrees'] = et

    # 5. KNN
    print("  训练 KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance',
                               metric='minkowski', p=2, n_jobs=-1)
    knn.fit(X_train, y_train)
    classifiers['knn'] = knn

    # 6. MLP
    print("  训练 MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                        max_iter=500, batch_size=64,
                        early_stopping=True, random_state=42)
    mlp.fit(X_train, y_train)
    classifiers['mlp'] = mlp

    for name, clf in classifiers.items():
        performances[name] = accuracy_score(y_val, clf.predict(X_val))
        print(f"    {name}: acc={performances[name]:.4f}")

    return classifiers, performances


def adaptive_weight_optimization(classifiers, X_val, y_val):
    """F1 自适应加权 — 对齐 3.1 adaptive_weight_optimization"""
    all_probs, f1_weights = {}, {}
    for name, clf in classifiers.items():
        all_probs[name] = clf.predict_proba(X_val)
        f1_weights[name] = max(f1_score(y_val, clf.predict(X_val), average='macro'), 0.01)

    total = sum(f1_weights.values())
    weights = {k: v / total for k, v in f1_weights.items()}

    weighted_probs = sum(weights[n] * all_probs[n] for n in classifiers)
    weighted_acc = accuracy_score(y_val, np.argmax(weighted_probs, axis=1))
    print(f"  加权投票准确率: {weighted_acc:.4f}")
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: weight={w:.4f}")
    return weights, weighted_acc


def create_stacking(classifiers, X_train, y_train, X_val, y_val):
    """Stacking 集成 — 对齐 3.1 create_advanced_stacking"""
    estimators = [(name, clf) for name, clf in classifiers.items()]
    meta = LogisticRegression(C=0.1, max_iter=2000, random_state=42)

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        passthrough=True,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    pred = stacking.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average='macro')
    print(f"  Stacking: acc={acc:.4f}  f1={f1:.4f}")
    print(classification_report(y_val, pred, target_names=['稳定', '轻微颤振', '严重颤振']))
    print("  混淆矩阵:")
    print(str(confusion_matrix(y_val, pred)))
    return stacking, acc, f1


# ==================== 训练管线 ====================

def train_pipeline(csv_path: str = DEFAULT_CSV):
    print("=" * 60)
    print("颤振诊断 Skill — 统一训练 v3.0 (SAE + 弱分类器 + Stacking)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 数据加载
    print(f"\n[1/7] 读取数据: {csv_path}")
    data = read_csv(csv_path)
    print(f"  采样点数: {data['n_samples']}, 采样率: {data['sampling_rate']:.1f}Hz")

    # 2. 分段
    print(f"\n[2/7] 分段 ({SEGMENT_SIZE}点/段, {int(OVERLAP_RATIO*100)}%重叠)...")
    segments, labels = segment_data(data)
    print(f"  分段数: {len(segments)}, 标签分布: {np.bincount(labels).tolist()}")

    # 3. 特征提取
    print(f"\n[3/7] 提取 50 维特征...")
    X_all = extract_all_features(segments, data['sampling_rate'])
    print(f"  特征矩阵: {X_all.shape}")

    # 4. 分层抽样 + 归一化 (2.2: StandardScaler)
    print(f"\n[4/7] 分层抽样 + 50维归一化...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(X_all, labels))
    X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    print(f"  训练集: {len(y_train)} {np.bincount(y_train, minlength=3).tolist()}, "
          f"验证集: {len(y_val)} {np.bincount(y_val, minlength=3).tolist()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # 5. SAE 降维 (2.2 方法: 50→512→256→128→16)
    print(f"\n[5/7] SAE 降维 (50→{ENCODING_DIM})...")
    sae, history = train_sae(X_train, X_val, encoding_dim=ENCODING_DIM,
                             epochs=300, batch_size=128, lr=0.001, patience=70)
    sae.eval()
    with torch.no_grad():
        X_train_enc = sae.get_encoded(torch.FloatTensor(X_train)).cpu().numpy()
        X_val_enc = sae.get_encoded(torch.FloatTensor(X_val)).cpu().numpy()

    # 信息保留率
    with torch.no_grad():
        _, X_train_recon = sae(torch.FloatTensor(X_train))
    mse = float(np.mean((X_train - X_train_recon.cpu().numpy()) ** 2))
    info_ret = (1 - mse / float(np.var(X_train))) * 100
    print(f"  信息保留率: {info_ret:.2f}%")

    # 6. 16维再标准化 (3.1 load_data 中的 StandardScaler)
    print(f"\n[6/7] 16 维编码特征再标准化...")
    scaler16 = StandardScaler()
    X_train_16 = scaler16.fit_transform(X_train_enc)
    X_val_16 = scaler16.transform(X_val_enc)

    # 7. 弱分类器 + F1 加权 + Stacking (3.1 方法)
    print(f"\n[7/7] 弱分类器 + Stacking...")
    classifiers, performances = train_base_classifiers(X_train_16, y_train, X_val_16, y_val)
    weights, weighted_acc = adaptive_weight_optimization(classifiers, X_val_16, y_val)
    stacking, stacking_acc, stacking_f1 = create_stacking(
        classifiers, X_train_16, y_train, X_val_16, y_val)

    # ==================== 保存 ====================
    print(f"\n保存模型到: {MODEL_DIR}")
    save_model(sae, str(MODEL_DIR), name='sae_model')
    joblib.dump(scaler, str(MODEL_DIR / 'scaler.pkl'))
    joblib.dump(scaler16, str(MODEL_DIR / 'scaler16.pkl'))

    fusion_data = {
        'classifiers': classifiers,
        'weights': weights,
        'stacking': stacking,
        'performances': performances,
        'weighted_acc': weighted_acc,
        'stacking_acc': stacking_acc,
        'stacking_f1': stacking_f1,
        'feature_names': ALL_FEATURE_NAMES,
        'segment_size': SEGMENT_SIZE,
        'encoding_dim': ENCODING_DIM,
    }
    joblib.dump(fusion_data, str(MODEL_DIR / 'fusion_model.pkl'))
    print("  sae_model.pth / scaler.pkl / scaler16.pkl / fusion_model.pkl 已保存")

    # 摘要
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  SAE 信息保留率:   {info_ret:.2f}%")
    print(f"  加权投票准确率:   {weighted_acc:.4f}")
    print(f"  Stacking 准确率: {stacking_acc:.4f}")
    print(f"  Stacking F1:     {stacking_f1:.4f}")

    return {
        'stacking_acc': stacking_acc, 'stacking_f1': stacking_f1,
        'weighted_acc': weighted_acc, 'info_retention': info_ret,
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    result = train_pipeline(csv_path)
    if result['stacking_f1'] > 0.8:
        print("\n✅ 训练成功！Stacking F1 > 0.8，模型可用。")
    else:
        print(f"\n⚠️  Stacking F1={result['stacking_f1']:.4f}，可能需要调整参数。")
    return result


if __name__ == '__main__':
    main()
