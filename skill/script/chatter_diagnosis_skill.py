"""
颤振诊断 Skill — 推理入口 v3.0（诊断 + 实时监控 双模式）

用法:
  python -m script.chatter_diagnosis_skill diagnose <signal.csv>   # 颤振诊断（三分类）
  python -m script.chatter_diagnosis_skill monitor  <signal.csv>   # 实时监控（z-score 基线偏离）

诊断流程（与训练管线严格一致）:
  0. 读取 CSV → 校验 → 立即绘制 4×1 原始信号图 (主轴/Z、X、Y 振动 + 三向力合力，X轴=时间) → <stem>_signal.png
  1. 分段(256点/段, 无重叠)
  2. 每段提取 50 维特征 (15*3 + 5)
  3. StandardScaler 归一化 (scaler.pkl)
  4. SAE 编码 50→16 (sae_model.pth, 2.2 方法)
  5. 16 维再标准化 (scaler16.pkl, 3.1 方法)
  6. 弱分类器 + Stacking 集成分类 (fusion_model.pkl, 3.1 方法)
  7. 输出诊断报告

监控流程:
  1. 读取 CSV → 分段(256点/段, 无重叠)
  2. 每段提取 50 维特征
  3. 与稳态基线 baseline_stats.json 比较，计算最大 z = |x_i − μ_i| / σ_i
  4. 按阈值评级 🟢<2.0 / 🟡2.0~3.5 / 🔴≥3.5，并比较前后半段趋势
  5. 输出监控报告
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime
from pathlib import Path
from collections import Counter

from script.feature_extractor import extract_50_features, ALL_FEATURE_NAMES
from script.train_sae_model import SparseAutoencoder
from script.signal_plot import plot_signals

# ==================== 路径与常量 ====================
SKILL_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = SKILL_DIR / 'script' / 'models'

SEGMENT_SIZE = 256          # 必须与训练一致
MIN_SAMPLES = 256           # 至少一个完整段
CLASS_NAMES = ['稳定加工', '空载', '颤振']

_model_cache = None


# ==================== 模型加载 ====================

def load_models(silent=False):
    """加载 SAE + 两级 Scaler + 弱分类器/Stacking"""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not silent:
        print("加载模型...")

    # 1. SAE (50→16)
    sae_path = MODEL_DIR / 'sae_model.pth'
    if not sae_path.exists():
        raise FileNotFoundError(
            f"SAE 模型不存在: {sae_path}\n请先运行 python -m script.train_all")
    ckpt = torch.load(str(sae_path), map_location='cpu', weights_only=False)
    sae = SparseAutoencoder(ckpt['input_dim'], ckpt['encoding_dim'])
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()

    # 2. 50维 Scaler
    scaler_path = MODEL_DIR / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler 不存在: {scaler_path}")
    scaler = joblib.load(str(scaler_path))

    # 3. 16维 Scaler
    scaler16_path = MODEL_DIR / 'scaler16.pkl'
    if not scaler16_path.exists():
        raise FileNotFoundError(f"Scaler16 不存在: {scaler16_path}\n请重新运行训练")
    scaler16 = joblib.load(str(scaler16_path))

    # 4. 弱分类器 + Stacking
    fusion_path = MODEL_DIR / 'fusion_model.pkl'
    if not fusion_path.exists():
        raise FileNotFoundError(f"融合模型不存在: {fusion_path}")
    fusion_data = joblib.load(str(fusion_path))

    _model_cache = {
        'sae': sae,
        'scaler': scaler,
        'scaler16': scaler16,
        'stacking': fusion_data['stacking'],
        'weights': fusion_data['weights'],
        'classifiers': fusion_data['classifiers'],
    }

    if not silent:
        print(f"  弱分类器: {list(fusion_data['classifiers'].keys())}")
        if 'stacking_f1' in fusion_data:
            print(f"  Stacking F1: {fusion_data['stacking_f1']:.4f}")

    return _model_cache


# ==================== 信号读取与分段 ====================

def _read_csv_with_encoding(csv_path):
    for enc in ('utf-8', 'gbk', 'gb18030', 'latin-1'):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(csv_path)


def read_and_validate_csv(csv_path):
    """读取并验证 CSV，前5列: 时间, X振动, Y振动, Z振动, 三向力合力"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = _read_csv_with_encoding(csv_path)
    if df.shape[1] < 5:
        raise ValueError(f"CSV列数不足 ({df.shape[1]}列)，需要至少5列: 时间, X, Y, Z, 力")

    cols = list(df.columns)
    time = pd.to_numeric(df[cols[0]], errors='coerce').values
    x = pd.to_numeric(df[cols[1]], errors='coerce').values
    y = pd.to_numeric(df[cols[2]], errors='coerce').values
    z = pd.to_numeric(df[cols[3]], errors='coerce').values
    force = pd.to_numeric(df[cols[4]], errors='coerce').values

    valid = ~(np.isnan(time) | np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(force))
    time, x, y, z, force = [arr[valid] for arr in (time, x, y, z, force)]

    n = len(time)
    if n < MIN_SAMPLES:
        return {'error': f'信号过短: {n}点 (最少需要{MIN_SAMPLES}点)'}

    dt = np.median(np.diff(time))
    fs = 1.0 / dt if dt > 0 else 1000.0

    return {'time': time, 'x': x, 'y': y, 'z': z, 'force': force,
            'n_samples': n, 'sampling_rate': fs}


def segment_signal(sig, segment_size=SEGMENT_SIZE):
    """无重叠分段（长度与训练一致）"""
    n = len(sig['time'])
    segments = []
    for start in range(0, n - segment_size + 1, segment_size):
        end = start + segment_size
        segments.append({
            'x': sig['x'][start:end], 'y': sig['y'][start:end],
            'z': sig['z'][start:end], 'force': sig['force'][start:end],
            't_start': sig['time'][start], 't_end': sig['time'][end - 1],
        })
    return segments


def _plot_signal_for(csv_path, sig):
    """为给定 CSV 生成 4×1 信号图，返回保存路径或 None。

    图与原始 CSV 同目录，命名 <stem>_signal.png。
    """
    csv_p = Path(csv_path)
    out = csv_p.with_name(csv_p.stem + '_signal.png')
    return plot_signals(sig, out_path=str(out))


# ==================== 诊断主函数 ====================

def diagnose_csv(csv_path, segment_size=SEGMENT_SIZE):
    """
    颤振诊断（单模式）：CSV → 50维特征 → 归一化 → SAE 16维 → 弱分类器 → Stacking

    Args:
        csv_path: 传感器 CSV 文件路径
        segment_size: 分段长度（默认256，与训练一致，不建议修改）

    Returns:
        诊断报告文本
    """
    models = load_models(silent=True)
    sae = models['sae']
    scaler = models['scaler']
    scaler16 = models['scaler16']
    stacking = models['stacking']
    classifiers = models['classifiers']
    weights = models['weights']

    sig = read_and_validate_csv(csv_path)
    if 'error' in sig:
        return f"⚠️ {sig['error']}"

    # 【v3.1】读入即绘制 4×1 原始信号图（不依赖模型，先于诊断报告生成）
    fig_path = _plot_signal_for(csv_path, sig)

    segments = segment_signal(sig, segment_size)
    if not segments:
        return "⚠️ 信号分段后为空"

    # 批量提取 50 维特征
    feats = np.array([
        extract_50_features(s['x'], s['y'], s['z'], s['force'],
                            sampling_rate=sig['sampling_rate'])
        for s in segments
    ])

    # 归一化 → SAE 16维 → 再标准化
    feats_scaled = scaler.transform(feats)
    with torch.no_grad():
        feats_16 = sae.get_encoded(torch.FloatTensor(feats_scaled)).cpu().numpy()
    feats_16 = scaler16.transform(feats_16)

    # 弱分类器加权投票（参考诊断） + Stacking（最终结论）
    weak_probs = sum(weights[name] * clf.predict_proba(feats_16)
                     for name, clf in classifiers.items())
    weak_preds = np.argmax(weak_probs, axis=1)

    stack_preds = stacking.predict(feats_16)
    stack_probs = stacking.predict_proba(feats_16)

    results = []
    for i, seg in enumerate(segments):
        results.append({
            't_start': seg['t_start'], 't_end': seg['t_end'],
            'weak_pred': int(weak_preds[i]),
            'prediction': int(stack_preds[i]),
            'confidence': float(np.max(stack_probs[i])),
        })

    # ==================== 报告 ====================
    rp = []
    rp.append("━" * 48)
    rp.append("  颤振诊断报告 (SAE + Stacking)")
    rp.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"  信号文件: {os.path.basename(csv_path)}")
    rp.append("━" * 48)

    rp.append("\n信号概况:")
    rp.append(f"  时长: {sig['time'][-1] - sig['time'][0]:.2f}s")
    rp.append(f"  采样点数: {sig['n_samples']}")
    rp.append(f"  采样率: {sig['sampling_rate']:.1f}Hz")
    rp.append(f"  分段数: {len(segments)} ({segment_size}点/段)")
    if fig_path:
        rp.append(f"  信号图: {fig_path}")
        rp.append("         (4×1 时序图：①主轴/Z振动 ②X振动 ③Y振动 ④三向力合力，X轴=时间)")

    rp.append("\n逐段诊断 (弱分类器投票 | Stacking最终):")
    for i, r in enumerate(results):
        rp.append(
            f"  段{i+1}: {r['t_start']:.2f}s-{r['t_end']:.2f}s  "
            f"投票={CLASS_NAMES[r['weak_pred']]} | "
            f"Stacking={CLASS_NAMES[r['prediction']]} "
            f"(置信度 {r['confidence']:.1%})"
        )

    counts = Counter(r['prediction'] for r in results)
    dominant = max(counts, key=counts.get)
    rp.append(f"\n整体结论: {CLASS_NAMES[dominant]}")
    rp.append("状态分布: " + ", ".join(
        f"{CLASS_NAMES[k]}×{v}" for k, v in sorted(counts.items())))

    if dominant == 2:
        rp.append("\n工艺建议: 检测到严重颤振 —")
        rp.append("  1. 立即降低进给速度 (15-30%)")
        rp.append("  2. 减小切削深度")
        rp.append("  3. 检查刀具磨损/破损")
        rp.append("  4. 调整主轴转速避开共振区")
    elif dominant == 1:
        rp.append("\n工艺建议: 检测到轻微颤振 — 建议适当降低进给速度并持续观察。")

    rp.append("\n" + "━" * 48)
    return '\n'.join(rp)


# ==================== 实时监控（z-score 基线偏离） ====================

def load_baseline():
    """加载稳态基线 (稳定工况 50 维 μ/σ)，返回按 ALL_FEATURE_NAMES 排序的 μ/σ 数组。"""
    base_path = MODEL_DIR / 'baseline_stats.json'
    if not base_path.exists():
        raise FileNotFoundError(
            f"监控基线不存在: {base_path}\n请先运行 python -m script.train_fusion_model")
    with open(base_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    mu = np.zeros(50)
    sigma = np.zeros(50)
    for i, name in enumerate(ALL_FEATURE_NAMES):
        entry = raw.get(f'稳定_{name}')
        if entry is None:
            raise KeyError(f"基线缺少特征: 稳定_{name}")
        mu[i] = float(entry['mean'])
        sigma[i] = float(entry['std'])
    return mu, sigma


def _monitor_rate(z):
    if z >= 3.5:
        return '🔴报警'
    if z >= 2.0:
        return '🟡关注'
    return '🟢正常'


def monitor_csv(csv_path, segment_size=SEGMENT_SIZE):
    """
    实时监控（z-score 基线偏离）：CSV → 50维特征 → 与稳态基线比较 → 逐段评级。

    段评分取 50 维特征中最大 z = |x_i − μ_i| / σ_i
    阈值: <2.0 🟢正常 / 2.0~3.5 🟡关注 / ≥3.5 🔴报警
    趋势: 后半段均 z vs 前半段均 z（>1.5× 为劣化，<0.5× 为好转，其余稳定）
    """
    sig = read_and_validate_csv(csv_path)
    if 'error' in sig:
        return f"⚠️ {sig['error']}"

    # 【v3.1】读入即绘制 4×1 原始信号图（先于监控报告生成）
    fig_path = _plot_signal_for(csv_path, sig)

    mu, sigma = load_baseline()
    segments = segment_signal(sig, segment_size)
    if not segments:
        return "⚠️ 信号分段后为空"

    feats = np.array([
        extract_50_features(s['x'], s['y'], s['z'], s['force'],
                            sampling_rate=sig['sampling_rate'])
        for s in segments
    ])  # (n_seg, 50)
    safe_sigma = np.where(sigma > 1e-9, sigma, 1e-9)
    z = np.abs(feats - mu) / safe_sigma          # (n_seg, 50)
    seg_score = z.max(axis=1)                     # 每段最大 z

    rp = []
    rp.append("━" * 48)
    rp.append("  颤振实时监控报告 (z-score 基线偏离)")
    rp.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"  信号文件: {os.path.basename(csv_path)}")
    rp.append("━" * 48)

    rp.append("\n信号概况:")
    rp.append(f"  时长: {sig['time'][-1] - sig['time'][0]:.2f}s")
    rp.append(f"  采样点数: {sig['n_samples']}")
    rp.append(f"  采样率: {sig['sampling_rate']:.1f}Hz")
    rp.append(f"  分段数: {len(segments)} ({segment_size}点/段)")
    if fig_path:
        rp.append(f"  信号图: {fig_path}")
        rp.append("         (4×1 时序图：①主轴/Z振动 ②X振动 ③Y振动 ④三向力合力，X轴=时间)")

    rp.append("\n逐段监控 (最大 z | 评级):")
    for i, seg in enumerate(segments):
        rp.append(
            f"  段{i+1}: {seg['t_start']:.2f}s-{seg['t_end']:.2f}s  "
            f"z={seg_score[i]:.2f} {_monitor_rate(seg_score[i])}"
        )

    n_red = int(np.sum(seg_score >= 3.5))
    n_yellow = int(np.sum((seg_score >= 2.0) & (seg_score < 3.5)))
    n_green = int(np.sum(seg_score < 2.0))
    if n_red > 0:
        status = '🔴报警'
    elif n_yellow > 0:
        status = '🟡关注'
    else:
        status = '🟢正常'

    if len(seg_score) >= 2:
        half = len(seg_score) // 2
        first_half = seg_score[:half].mean()
        second_half = seg_score[half:].mean()
        if second_half > first_half * 1.5:
            trend = '📈劣化'
        elif second_half < first_half * 0.5:
            trend = '📉好转'
        else:
            trend = '↔️稳定'
        trend_detail = f"  (前半均z={first_half:.2f}, 后半均z={second_half:.2f})"
    else:
        trend = '↔️稳定'
        trend_detail = '  (仅单段，无趋势)'

    rp.append("\n整体评估:")
    rp.append(f"  最大 z: {seg_score.max():.2f}   平均 z: {seg_score.mean():.2f}")
    rp.append(f"  状态: {status}  (🟢{n_green} / 🟡{n_yellow} / 🔴{n_red})")
    rp.append(f"  趋势: {trend}{trend_detail}")

    if n_red > 0:
        rp.append("\n建议: 检测到报警段 — 立即检查加工状态，降低进给速度、减小切深、避开共振区。")
    elif n_yellow > 0:
        rp.append("\n建议: 出现关注段 — 持续观察，适当降低进给速度。")
    else:
        rp.append("\n建议: 全程正常，无需干预。")
    rp.append("\n" + "━" * 48)
    return '\n'.join(rp)


# ==================== 命令行入口 ====================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python -m script.chatter_diagnosis_skill [diagnose|monitor] <signal.csv>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd in ('diagnose', 'monitor'):
        if len(sys.argv) < 3:
            print("错误: 缺少信号 CSV 路径")
            sys.exit(1)
        csv_path = sys.argv[2]
    else:
        cmd, csv_path = 'diagnose', sys.argv[1]
    print(diagnose_csv(csv_path) if cmd == 'diagnose' else monitor_csv(csv_path))
