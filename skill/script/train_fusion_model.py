"""
监控基线构建脚本 (稳定工况 50 维特征 μ/σ)

产出 script/models/baseline_stats.json，供实时监控模式 (monitor_csv) 使用：
对原始训练数据分段提取 50 维特征，筛选「稳定」工况段，
计算每维特征的均值 μ 与标准差 σ，键形如 稳定_X_Clearance_Factor。

注：颤振诊断所需的融合模型 fusion_model.pkl / scaler16.pkl 由
    script/train_all.py（一键诊断训练）产出；本脚本仅负责监控基线。

用法:
  python -m script.train_fusion_model [训练数据.csv]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from script.feature_extractor import extract_50_features, ALL_FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parent.parent / 'script' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CSV = r"C:\Users\siat\Desktop\平衡后_三种工况数据.csv"
SEGMENT_SIZE = 256


def _label_to_int(s):
    s = str(s).strip()
    if '稳定' in s or '正常' in s:
        return 0
    if '空载' in s or '轻微' in s or '轻度' in s:
        return 1
    if '颤振' in s or '严重' in s or '重度' in s:
        return 2
    return -1


def read_segments(csv_path):
    """读取原始训练 CSV → 每段 50 维特征 + 标签(0稳定/1轻微/2严重)。"""
    df = pd.read_csv(csv_path, encoding='utf-8', header=None)
    df = df.iloc[1:].copy().reset_index(drop=True)
    for i in range(5):
        df[i] = pd.to_numeric(df[i], errors='coerce')
    time = df[0].values
    x = df[1].values
    y = df[2].values
    z = df[3].values
    force = df[4].values
    labels_raw = df[5].values if df.shape[1] >= 6 else None

    valid = ~(np.isnan(time) | np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(force))
    time, x, y, z, force = time[valid], x[valid], y[valid], z[valid], force[valid]
    if labels_raw is not None:
        labels = np.array([_label_to_int(v) for v in labels_raw[valid]])
        valid &= (labels >= 0)
        time, x, y, z, force = time[valid], x[valid], y[valid], z[valid], force[valid]
        labels = labels[valid]
    else:
        labels = None

    dt = np.median(np.diff(time)) if len(time) > 1 else 1e-3
    fs = 1.0 / dt if dt > 0 else 1000.0

    n = len(time)
    feats, seg_labels = [], []
    for start in range(0, n - SEGMENT_SIZE + 1, SEGMENT_SIZE):
        seg = extract_50_features(
            x[start:start + SEGMENT_SIZE], y[start:start + SEGMENT_SIZE],
            z[start:start + SEGMENT_SIZE], force[start:start + SEGMENT_SIZE],
            sampling_rate=fs)
        feats.append(seg)
        if labels is not None:
            seg_labels.append(labels[start + SEGMENT_SIZE // 2])
    return np.array(feats), (np.array(seg_labels) if seg_labels else None)


def build_baseline(csv_path):
    """构建并保存稳定工况 50 维特征的 μ/σ 基线。"""
    feats, labels = read_segments(csv_path)
    if labels is None:
        raise ValueError("训练 CSV 缺少工况标签列，无法构建稳态基线")
    mask = (labels == 0)
    if int(mask.sum()) == 0:
        raise ValueError("未找到「稳定」工况样本，无法构建基线")
    stable = feats[mask]
    baseline = {}
    for i, name in enumerate(ALL_FEATURE_NAMES):
        col = stable[:, i]
        baseline[f'稳定_{name}'] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
        }
    out = MODEL_DIR / 'baseline_stats.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"稳态基线已保存: {out}")
    print(f"  稳定段数: {int(mask.sum())} / 总段数: {len(feats)}")
    print(f"  特征数: {len(ALL_FEATURE_NAMES)}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print("=" * 60)
    print("监控基线构建 (稳定工况 50 维 μ/σ)")
    print("=" * 60)
    build_baseline(csv_path)
