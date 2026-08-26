"""
决策融合模型训练脚本 (基于固定50维特征+SAE编码)
加载SAE输出的16维编码特征，训练集成分类器。

输出: models/fusion_model.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 从独立模块导入 FusionModel，确保序列化/反序列化时类引用一致
try:
    from .fusion_model_def import FusionModel
except ImportError:
    from fusion_model_def import FusionModel


def train_fusion():
    print("=" * 60)
    print("决策融合模型训练")
    print("=" * 60)

    # 1. 加载SAE编码特征
    encoded_path = os.path.join(MODEL_DIR, 'sae_encoded_features.csv')
    if not os.path.exists(encoded_path):
        print(f"错误: 找不到 {encoded_path}")
        print("请先运行 train_sae_model.py")
        return

    df = pd.read_csv(encoded_path)
    feature_cols = [c for c in df.columns if c.startswith('SAE_')]
    X = df[feature_cols].values
    y = df['label'].values

    # 划分训练/验证 (按索引前90%训练, 后10%验证)
    n = len(X)
    split = int(n * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"训练标签分布: {np.bincount(y_train.astype(int))}")
    print(f"验证标签分布: {np.bincount(y_val.astype(int))}")

    # 2. 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 3. 训练基分类器
    classifiers = {}

    # LightGBM
    print("\n训练 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.0, min_child_samples=20,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train_s, y_train)
    classifiers['lightgbm'] = lgb_model
    print(f"  准确率: {accuracy_score(y_val, lgb_model.predict(X_val_s)):.4f}")

    # CatBoost
    if HAS_CATBOOST:
        print("训练 CatBoost...")
        cat = CatBoostClassifier(
            iterations=800, depth=8, learning_rate=0.05,
            verbose=False, random_seed=42,
        )
        cat.fit(X_train_s, y_train)
        classifiers['catboost'] = cat
        print(f"  准确率: {accuracy_score(y_val, cat.predict(X_val_s)):.4f}")

    # SVM
    print("训练 SVM...")
    svm = SVC(C=50, kernel='rbf', gamma='scale', probability=True, cache_size=2000)
    svm.fit(X_train_s, y_train)
    classifiers['svm'] = svm
    print(f"  准确率: {accuracy_score(y_val, svm.predict(X_val_s)):.4f}")

    # ExtraTrees
    print("训练 ExtraTrees...")
    et = ExtraTreesClassifier(
        n_estimators=300, max_features='sqrt',
        random_state=42, n_jobs=-1,
    )
    et.fit(X_train_s, y_train)
    classifiers['extratrees'] = et
    print(f"  准确率: {accuracy_score(y_val, et.predict(X_val_s)):.4f}")

    # KNN
    print("训练 KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train_s, y_train)
    classifiers['knn'] = knn
    print(f"  准确率: {accuracy_score(y_val, knn.predict(X_val_s)):.4f}")

    # MLP
    print("训练 MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        max_iter=500, batch_size=64, early_stopping=True,
    )
    mlp.fit(X_train_s, y_train)
    classifiers['mlp'] = mlp
    print(f"  准确率: {accuracy_score(y_val, mlp.predict(X_val_s)):.4f}")

    # 4. 计算F1权重
    print("\n计算F1权重...")
    f1_weights = {}
    for name, clf in classifiers.items():
        pred = clf.predict(X_val_s)
        f1 = max(f1_score(y_val, pred, average='macro'), 0.01)
        f1_weights[name] = f1
    f1_total = sum(f1_weights.values())
    weights = {k: v / f1_total for k, v in f1_weights.items()}
    for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {w:.4f}")

    # 5. Stacking
    print("\n训练 Stacking...")
    estimators = [(name, clf) for name, clf in classifiers.items()]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        passthrough=True,
        n_jobs=-1,
    )
    stacking.fit(X_train_s, y_train)
    stacking_acc = accuracy_score(y_val, stacking.predict(X_val_s))
    print(f"  Stacking准确率: {stacking_acc:.4f}")

    print("\n分类报告:")
    report = classification_report(y_val, stacking.predict(X_val_s),
                                   target_names=['稳定', '轻微颤振', '严重颤振'])
    print(report)

    # 6. 保存融合模型
    fusion = FusionModel()
    fusion.scaler = scaler
    fusion.classifiers = classifiers
    fusion.f1_weights = weights
    fusion.stacking = stacking

    model_path = os.path.join(MODEL_DIR, 'fusion_model.pkl')
    joblib.dump(fusion, model_path)
    print(f"\n融合模型已保存: {model_path}")

    # 保存基线特征统计 (用于解释)
    baseline = {}
    for i in range(len(feature_cols)):
        col = feature_cols[i]
        for cls_name, cls_label in [('稳定', 0), ('轻微颤振', 1), ('严重颤振', 2)]:
            mask = y_train.astype(int) == cls_label
            baseline[f"{cls_name}_{col}"] = {
                'mean': float(X_train[mask, i].mean()),
                'std': float(X_train[mask, i].std()),
            }

    baseline_path = os.path.join(MODEL_DIR, 'baseline_stats.json')
    import json
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2)
    print(f"基线统计已保存: {baseline_path}")

    return fusion


if __name__ == '__main__':
    train_fusion()
