"""
颤振诊断 Skill - 主入口

输入: CSV文件路径 (列: 时间, 主轴振动, X轴振动, Y轴振动, 三向力合力)
输出: 可解释的诊断报告

使用方式:
    from chatter_diagnosis_skill import diagnose_csv
    report = diagnose_csv("path/to/signal.csv")
    print(report)
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import joblib

# 与训练脚本保持一致
from .feature_extractor import extract_50_features, ALL_FEATURE_NAMES, VIB_FEATURE_NAMES, FORCE_FEATURE_NAMES
from .train_sae_model import SparseAutoencoder

warnings.filterwarnings('ignore')

# ==================== 路径与配置 ====================
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_MODULE_DIR, 'models')

# 与训练标签严格一致 (skill/script/train_all.py: 稳定加工→0, 空载→1, 颤振→2)
CLASS_NAMES = ['稳定加工', '空载', '颤振']
SEGMENT_SIZE = 256          # 必须与训练一致 (skill/script/train_all.py 按 256 点/段分段)
MIN_SAMPLES = 256           # 至少一个完整段

# 先验规则: 用于生成可解释的诊断依据
# 键: 特征名, 值: (偏离方向, 工艺建议)
# 方向: + 表示增大→颤振, - 表示减小→异常
VIB_RULES = {
    'RMS': ('增大', '降低进给速度'),
    'Peak': ('增大', '检查刀具磨损'),
    'Peak_to_Peak': ('增大', '降低切削深度'),
    'Signal_Energy': ('增大', '减小主轴转速'),
    'Spectral_Energy': ('增大', '检查工艺系统刚性'),
    'STFT_Total_Energy': ('增大', '降低进给速度'),
    'Time_Frequency_Entropy': ('增大', '调整切削参数'),
    'Power_Spectrum_Peak': ('增大', '避开共振频率'),
    'Frequency_Variance': ('增大', '调整转速避开谐振区'),
    'Clearance_Factor': ('增大', '检查刀具状态'),
    'Std': ('增大', '降低切削参数'),
    'Variance': ('增大', '降低进给速度'),
    'Shape_Factor': ('变化', '检查系统稳定性'),
    'STFT_Mean': ('增大', '降低切削参数'),
    'Power_Spectrum_Clearance': ('增大', '检查刀具磨损'),
}

FORCE_RULES = {
    'Force_Peak2Peak': ('增大', '降低进给速度'),
    'Force_Peak': ('增大', '降低切削深度'),
    'Force_Crest_Factor': ('增大', '检查刀具状态'),
    'Force_Impulse_Factor': ('增大', '降低进给速度'),
    'Force_Freq_Variance': ('增大', '调整转速'),
}


# ==================== 模型加载 ====================
_model_cache = {}


def _fix_sklearn_compat(obj, _seen=None):
    """跨 sklearn 版本兼容补丁。

    模型 pickle 与当前运行环境的 sklearn 版本不一致时，
    LogisticRegression 可能缺失 multi_class 属性，导致
    predict_proba 抛 AttributeError。这里递归遍历集成模型，
    为缺失属性的 LogisticRegression 补上 'auto'（softmax 路径，
    与训练时多分类行为一致）。
    """
    from sklearn.linear_model import LogisticRegression
    if _seen is None:
        _seen = set()
    if obj is None or id(obj) in _seen:
        return
    _seen.add(id(obj))

    if isinstance(obj, LogisticRegression):
        if not hasattr(obj, 'multi_class'):
            obj.multi_class = 'auto'
        return

    for attr in ('final_estimator_', 'final_estimator', 'estimators_', 'estimators',
                 'named_estimators_', 'base_estimator_', 'estimator_', 'estimator', 'steps'):
        try:
            sub = getattr(obj, attr, None)
        except Exception:
            continue
        if sub is None:
            continue
        if isinstance(sub, dict):
            for v in sub.values():
                _fix_sklearn_compat(v, _seen)
        elif isinstance(sub, (list, tuple)):
            for item in sub:
                if isinstance(item, tuple):
                    for x in item:
                        if not isinstance(x, str):
                            _fix_sklearn_compat(x, _seen)
                elif not isinstance(item, str):
                    _fix_sklearn_compat(item, _seen)
        else:
            _fix_sklearn_compat(sub, _seen)


def load_models(silent=False):
    """加载 SAE(50→16) + 两级 Scaler + 弱分类器/Stacking 融合模型。

    模型由 skill/script/train_all.py 训练生成，与训练管线严格一致：
      scaler.pkl(50维) → sae_model.pth(50→16) → scaler16.pkl(16维) → fusion_model.pkl(Stacking)
    """
    if _model_cache:
        return _model_cache

    if not silent:
        print("加载模型...")

    # 1. SAE (50→16)
    sae_path = os.path.join(_MODEL_DIR, 'sae_model.pth')
    if not os.path.exists(sae_path):
        raise FileNotFoundError(
            f"找不到 SAE 模型: {sae_path}\n请先运行 python -m skill.script.train_all 训练")
    ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
    sae = SparseAutoencoder(ckpt['input_dim'], ckpt['encoding_dim'])
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()

    # 2. 50维 Scaler
    scaler_path = os.path.join(_MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 Scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 3. 16维 Scaler
    scaler16_path = os.path.join(_MODEL_DIR, 'scaler16.pkl')
    if not os.path.exists(scaler16_path):
        raise FileNotFoundError(f"找不到 Scaler16: {scaler16_path}\n请重新运行训练")
    scaler16 = joblib.load(scaler16_path)

    # 4. 弱分类器 + Stacking
    fusion_path = os.path.join(_MODEL_DIR, 'fusion_model.pkl')
    if not os.path.exists(fusion_path):
        raise FileNotFoundError(f"找不到融合模型: {fusion_path}")
    fusion_data = joblib.load(fusion_path)
    stacking = fusion_data['stacking']
    classifiers = fusion_data['classifiers']
    weights = fusion_data.get('weights', {})

    # sklearn 跨版本兼容补丁（修复 predict_proba 的 multi_class AttributeError）
    _fix_sklearn_compat(stacking)
    for clf in (classifiers.values() if isinstance(classifiers, dict) else []):
        _fix_sklearn_compat(clf)

    # Baseline stats (用于规则模式分析，可选)
    baseline_path = os.path.join(_MODEL_DIR, 'baseline_stats.json')
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding='utf-8') as f:
            baseline = json.load(f)

    _model_cache['sae'] = sae
    _model_cache['scaler'] = scaler
    _model_cache['scaler16'] = scaler16
    _model_cache['stacking'] = stacking
    _model_cache['classifiers'] = classifiers
    _model_cache['weights'] = weights
    _model_cache['baseline'] = baseline

    if not silent:
        print(f"  模型加载完成: SAE(50→{ckpt['encoding_dim']}) + Stacking({list(classifiers.keys())})")
    return _model_cache


# ==================== 信号处理 ====================
def _read_csv_with_encoding(csv_path: str, **kwargs):
    """尝试多种编码读取 CSV，优先 UTF-8，回退 GBK / GB18030。"""
    for enc in ('utf-8', 'gbk', 'gb18030', 'latin-1'):
        try:
            return pd.read_csv(csv_path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最后尝试不带编码参数（走 Python 默认）
    return pd.read_csv(csv_path, **kwargs)


def read_and_validate_csv(csv_path: str) -> dict:
    """
    读取并验证CSV文件。
    自动探测编码（UTF-8 / GBK / GB18030），
    兼容部分系统导出的首列行号/索引列。

    Returns:
        dict with keys: time, 主轴, X, Y, force_mag
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    # 读取 header 行与第一行数据，比对列数
    df = _read_csv_with_encoding(csv_path)
    n_header = len(df.columns)

    with open(csv_path, 'rb') as f:
        raw = f.read(4096)
    # 用换行符分割取第二行（第一条数据）
    lines = raw.split(b'\n')
    if len(lines) >= 2:
        n_data = len(lines[1].split(b','))
    else:
        n_data = n_header

    # 如果 header 列数 > 数据列数（常见：系统导出时 header 多一个前导行号列）
    if n_header > n_data >= 1:
        df = _read_csv_with_encoding(csv_path, header=None, skiprows=1)

    if df.shape[1] < 5:
        raise ValueError(f"CSV列数不足 ({df.shape[1]}列)，需要5列: 时间, 主轴振动, X轴振动, Y轴振动, 三向力合力")

    # 取前5列
    cols = [df.columns[i] for i in range(5)]
    time_col = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    vib主轴 = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    vib_x = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
    vib_y = pd.to_numeric(df.iloc[:, 3], errors='coerce').values
    force_col = pd.to_numeric(df.iloc[:, 4], errors='coerce').values

    # 去除NaN
    valid = ~np.isnan(time_col) & ~np.isnan(vib主轴) & ~np.isnan(vib_x) & ~np.isnan(vib_y) & ~np.isnan(force_col)
    time_col = time_col[valid]
    vib主轴 = vib主轴[valid]
    vib_x = vib_x[valid]
    vib_y = vib_y[valid]
    force_col = force_col[valid]

    n = len(time_col)
    if n < MIN_SAMPLES:
        return {'error': f'信号过短: {n}个点 (最小需要 {MIN_SAMPLES} 个点). 建议增加采样时间。'}

    duration = time_col[-1] - time_col[0] if len(time_col) > 1 else 0
    dt = np.median(np.diff(time_col))
    fs = 1.0 / dt if dt > 0 else 1000.0

    return {
        'time': time_col,
        '主轴': vib主轴,
        'X': vib_x,
        'Y': vib_y,
        'force': force_col,
        'n_samples': n,
        'duration': duration,
        'sampling_rate': fs,
    }


def segment_signal(time, vib主轴, vib_x, vib_y, force, segment_size=SEGMENT_SIZE):
    """
    将长信号无重叠分段（长度与训练一致：256 点/段，丢弃尾部不足一段的部分）。

    Returns:
        list of dicts with segment data
    """
    n = len(time)
    segments = []
    for start in range(0, n - segment_size + 1, segment_size):
        end = start + segment_size
        segments.append({
            'time': time[start:end],
            '主轴': vib主轴[start:end],
            'X': vib_x[start:end],
            'Y': vib_y[start:end],
            'force': force[start:end],
            't_start': time[start],
            't_end': time[end - 1],
        })
    return segments


def diagnose_segment(vib主轴, vib_x, vib_y, force, fs, models):
    """诊断单个信号段（SAE 50→16 + Stacking 融合，与训练管线严格一致）。"""
    # 提取50维特征
    features_50 = extract_50_features(vib主轴, vib_x, vib_y, force, sampling_rate=fs)

    # 归一化 → SAE 16维 → 再标准化
    scaler = models['scaler']
    sae = models['sae']
    scaler16 = models['scaler16']
    stacking = models['stacking']

    feats_scaled = scaler.transform(features_50.reshape(1, -1))
    with torch.no_grad():
        feats_16 = sae.get_encoded(torch.FloatTensor(feats_scaled)).cpu().numpy()
    feats_16 = scaler16.transform(feats_16)

    # Stacking 集成分类
    pred = int(stacking.predict(feats_16)[0])
    try:
        proba = stacking.predict_proba(feats_16)[0]
    except Exception:
        # 版本兼容问题时降级：用 predict 结果构造独热概率，保证诊断不中断
        proba = np.zeros(len(CLASS_NAMES))
        proba[pred] = 1.0
    confidence = float(np.max(proba))

    return {
        'prediction': pred,
        'class_name': CLASS_NAMES[pred],
        'confidence': confidence,
        'probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        'features_50': features_50,
    }


# ==================== 诊断报告生成 ====================
def generate_description(signal_info):
    """生成信号整体特征描述."""
    lines = []
    duration = signal_info['duration']
    n = signal_info['n_samples']
    fs = signal_info['sampling_rate']
    lines.append(f"信号时长: {duration:.2f}s, 采样点数: {n}, 采样率: {fs:.1f}Hz")

    # 各通道统计
    for name, sig in [('主轴振动', signal_info['主轴']),
                       ('X轴振动', signal_info['X']),
                       ('Y轴振动', signal_info['Y']),
                       ('三向力合力', signal_info['force'])]:
        lines.append(f"  {name}: 均值={np.mean(np.abs(sig)):.4f}, "
                      f"标准差={np.std(sig):.4f}, "
                      f"峰值={np.max(np.abs(sig)):.4f}")
    return '\n'.join(lines)


def generate_explanation(features_50, models, prediction):
    """基于特征值和基线对比生成可解释的依据."""
    lines = []
    baseline = models.get('baseline', {})

    deviations = _collect_deviations(features_50, baseline, None)

    if deviations:
        deviations.sort(key=lambda x: abs(x[4]), reverse=True)
        lines.append("关键特征偏离分析 (与稳定加工基线对比):")
        for name, fv, mean, std, z in deviations[:5]:
            direction = "升高" if fv > mean else "降低"
            lines.append(f"  - {name}: 当前值={fv:.4f}, 基线={mean:.4f}±{std:.4f}, "
                         f"偏离{direction} (z={abs(z):.2f})")
    else:
        lines.append("关键特征分析 (基线缺失，仅列当前值):")
        for idx, name in enumerate(ALL_FEATURE_NAMES[:10]):
            lines.append(f"  - {name}: {features_50[idx]:.4f}")

    return '\n'.join(lines)


def generate_recommendation(prediction, features_50, models):
    """根据诊断结果生成工艺建议."""
    recommendations = {
        0: ("当前为稳定加工状态，建议保持现有切削参数。", []),
        1: ("检测到空载运行（无切削负载），建议:", [
            "确认是否处于换刀/空走刀阶段，属正常工况",
            "若应在加工中，请检查刀具与工件是否正常接触",
            "检查力传感器信号是否正常",
        ]),
        2: ("检测到颤振，建议:", [
            "立即降低进给速度 (15-30%)",
            "减小切削深度",
            "检查刀具是否磨损或破损",
            "考虑调整主轴转速避开共振区",
            "检查工件夹持刚性",
        ]),
    }

    base_text, actions = recommendations.get(int(prediction), ("诊断结果未知", []))

    specific_advice = []
    # 根据最偏离的特征补充针对性建议 (仅颤振时)
    baseline = models.get('baseline', {})
    all_rules = {**VIB_RULES, **{k.replace('Force_', ''): v for k, v in FORCE_RULES.items()}}
    for name, fv, bm, bs, z in _collect_deviations(features_50, baseline, None):
        if z > 2.0:  # 仅正向(高于基线)偏离触发工艺建议
            short = name.split('_', 1)[-1] if '_' in name else name
            if short in all_rules:
                _, advice = all_rules[short]
                if advice not in specific_advice:
                    specific_advice.append(advice)

    if specific_advice and int(prediction) == 2:
        for adv in specific_advice[:2]:
            if adv not in actions:
                actions.append(adv)

    return base_text + '\n' + '\n'.join(f"  {i+1}. {a}" for i, a in enumerate(actions))


# ==================== 多模式诊断逻辑 ====================

VIB_FEATURE_NAMES_LIST = VIB_FEATURE_NAMES  # 从 feature_extractor 导入

# 特征类别集合（按 v3.0 特征名后缀分类，用于各规则模式选取特征）
_AMPLITUDE_FEATS = {'RMS', 'Peak', 'Peak_to_Peak', 'Std', 'Variance',
                    'Signal_Energy', 'Mean', 'Peak2Peak', 'Kurtosis', 'Skewness'}
_FREQUENCY_FEATS = {'Power_Spectrum_Peak', 'Power_Spectrum_Clearance',
                    'Frequency_Variance', 'Spectral_Energy', 'Spectral_Centroid',
                    'Mean_Square_Frequency', 'Freq_Variance', 'Peak_Count'}
_TIMEFREQ_FEATS = {'STFT_Mean', 'STFT_Total_Energy', 'Time_Frequency_Entropy'}


def _collect_deviations(features_50, baseline, feat_filter=None):
    """按特征名匹配基线 (baseline_stats.json 的 稳定_{特征名} 键)，
    返回 [(特征名, 当前值, 基线均值, 基线std, 带符号z值), ...]。

    z > 0 表示高于稳定加工基线（颤振倾向），z < 0 表示低于基线（空载倾向）。
    """
    deviations = []
    for idx, name in enumerate(ALL_FEATURE_NAMES):
        if feat_filter is not None:
            _, _, feat = name.partition('_')
            if feat not in feat_filter:
                continue
        key = f"稳定_{name}"
        if key in baseline:
            bm = baseline[key]['mean']
            bs = baseline[key]['std']
            if bs > 0:
                z = (features_50[idx] - bm) / bs
                deviations.append((name, features_50[idx], bm, bs, z))
    return deviations


def _decide_from_deviations(deviations, z_chatter=2.5, z_stable=1.5):
    """根据带符号偏离决定类别:
      最大|z| <= z_stable        → 0 稳定加工
      主导偏离为正且超阈值        → 2 颤振
      主导偏离为负且超阈值        → 1 空载
    返回 (pred, max_abs_z, top_deviations)
    """
    top = sorted(deviations, key=lambda d: abs(d[4]), reverse=True)[:5]
    max_abs = abs(top[0][4]) if top else 0.0
    mean_top = float(np.mean([d[4] for d in top])) if top else 0.0

    if max_abs <= z_stable:
        return 0, max_abs, top
    if mean_top >= 0 and max_abs > z_chatter:
        return 2, max_abs, top
    if mean_top < 0:
        return 1, max_abs, top
    # 偏高但未达颤振阈值 → 视为稳定加工
    return 0, max_abs, top


def _make_probabilities(pred, confidence):
    """按预测类别构造概率字典（规则模式为近似值）。"""
    proba = {n: round((1.0 - confidence) / 2, 3) for n in CLASS_NAMES}
    proba[CLASS_NAMES[pred]] = round(confidence, 3)
    return proba


def diagnose_amplitude_mode(features_50, models) -> dict:
    """
    幅值阈值模式：基于 RMS/峰值等时域幅值特征与稳定加工基线对比。
    显著高于基线 → 颤振；显著低于基线 → 空载。
    """
    baseline = models.get('baseline', {})
    deviations = _collect_deviations(features_50, baseline, _AMPLITUDE_FEATS)

    if not deviations:
        return {'prediction': 0, 'class_name': CLASS_NAMES[0], 'confidence': 0.5,
                'probabilities': _make_probabilities(0, 0.5),
                'features_50': features_50}

    pred, max_abs, top = _decide_from_deviations(deviations, z_chatter=3.5, z_stable=1.5)
    conf = min(0.9, 0.5 + max_abs * 0.08) if pred != 0 else max(0.6, 1.0 - max_abs * 0.15)

    return {
        'prediction': pred,
        'class_name': CLASS_NAMES[pred],
        'confidence': conf,
        'probabilities': _make_probabilities(pred, conf),
        'features_50': features_50,
        'key_deviations': top,
    }


def diagnose_frequency_mode(features_50, models) -> dict:
    """
    频谱特征模式：基于功率谱峰值、频率方差等频域特征与稳定加工基线对比。
    """
    baseline = models.get('baseline', {})
    deviations = _collect_deviations(features_50, baseline, _FREQUENCY_FEATS)

    if not deviations:
        return {'prediction': 0, 'class_name': CLASS_NAMES[0], 'confidence': 0.5,
                'probabilities': _make_probabilities(0, 0.5),
                'features_50': features_50}

    pred, max_abs, top = _decide_from_deviations(deviations, z_chatter=3.5, z_stable=1.5)
    conf = min(0.9, 0.5 + max_abs * 0.08) if pred != 0 else max(0.6, 1.0 - max_abs * 0.15)

    return {
        'prediction': pred,
        'class_name': CLASS_NAMES[pred],
        'confidence': conf,
        'probabilities': _make_probabilities(pred, conf),
        'features_50': features_50,
        'key_deviations': top,
    }


def diagnose_timefreq_mode(features_50, models) -> dict:
    """
    时频分析模式：基于 STFT 时频谱能量/熵与稳定加工基线对比。
    """
    baseline = models.get('baseline', {})
    deviations = _collect_deviations(features_50, baseline, _TIMEFREQ_FEATS)

    if not deviations:
        return {'prediction': 0, 'class_name': CLASS_NAMES[0], 'confidence': 0.5,
                'probabilities': _make_probabilities(0, 0.5),
                'features_50': features_50}

    pred, max_abs, top = _decide_from_deviations(deviations, z_chatter=3.5, z_stable=1.5)
    conf = min(0.9, 0.4 + max_abs * 0.08) if pred != 0 else max(0.55, 1.0 - max_abs * 0.12)

    return {
        'prediction': pred,
        'class_name': CLASS_NAMES[pred],
        'confidence': conf,
        'probabilities': _make_probabilities(pred, conf),
        'features_50': features_50,
        'key_deviations': top,
    }


def diagnose_trend_mode(features_50, models) -> dict:
    """
    趋势监测模式：综合全部 50 维特征的偏离程度与方向，侧重发展态势。
    """
    baseline = models.get('baseline', {})
    all_deviations = _collect_deviations(features_50, baseline, None)

    if not all_deviations:
        return {'prediction': 0, 'class_name': CLASS_NAMES[0], 'confidence': 0.5,
                'probabilities': _make_probabilities(0, 0.5),
                'features_50': features_50}

    zs = [d[4] for d in all_deviations]
    abs_zs = [abs(z) for z in zs]
    mean_z = float(np.mean(abs_zs))
    max_z = float(max(abs_zs))
    pct_high = sum(1 for z in abs_zs if z > 2.0) / len(abs_zs)
    # 偏离方向：高偏离特征中正向(高于基线)与负向(低于基线)的净方向
    high_signed = [z for z in zs if abs(z) > 1.5]
    net_direction = float(np.mean(high_signed)) if high_signed else 0.0

    # 趋势评分：平均偏离 + 峰值偏离 + 高偏离比例
    trend_score = mean_z * 0.3 + max_z * 0.5 + pct_high * 3.0

    if trend_score <= 1.2:
        pred = 0
        conf = max(0.55, 1.0 - trend_score * 0.2)
    elif net_direction < 0:
        pred = 1  # 整体显著低于稳定加工基线 → 空载
        conf = min(0.85, 0.4 + trend_score * 0.15)
    elif trend_score > 2.5:
        pred = 2  # 整体显著高于基线且偏离剧烈 → 颤振
        conf = min(0.90, trend_score * 0.25)
    else:
        pred = 0
        conf = 0.55

    return {
        'prediction': pred,
        'class_name': CLASS_NAMES[pred],
        'confidence': conf,
        'probabilities': _make_probabilities(pred, conf),
        'features_50': features_50,
        'trend_score': trend_score,
        'mean_deviation': mean_z,
        'max_deviation': max_z,
        'high_deviation_ratio': pct_high,
    }


# ==================== 综合诊断（多模式融合） ====================
_MODE_LIST = ["chatter_amplitude", "chatter_frequency", "chatter_timefreq", "chatter_trend"]

def _run_single_mode(features_50, mode: str, models) -> dict:
    """运行单一规则模式，返回诊断结果"""
    func_map = {
        "chatter_amplitude": diagnose_amplitude_mode,
        "chatter_frequency": diagnose_frequency_mode,
        "chatter_timefreq": diagnose_timefreq_mode,
        "chatter_trend": diagnose_trend_mode,
    }
    func = func_map.get(mode)
    if func:
        return func(features_50, models)
    return None


def diagnose_comprehensive(csv_path: str, long_segment_size: int = SEGMENT_SIZE) -> str:
    """
    综合诊断：时域分段 → 逐段诊断 → 简洁报告。
    报告格式匹配用户模板:
      【诊断概览】总时长 + 工况划分
      重点段分析 + 关键特征偏离
      【工艺建议】
    """
    models = load_models()
    result = read_and_validate_csv(csv_path)
    if 'error' in result:
        return f"⚠️ {result['error']}"

    sig = result
    segments = segment_signal(
        sig['time'], sig['主轴'], sig['X'], sig['Y'], sig['force'],
        segment_size=long_segment_size,
    )

    from collections import Counter
    duration = sig['time'][-1] - sig['time'][0]

    # ========== 逐段诊断 ==========
    segment_results = []
    all_preds = []
    worst_pred = 0
    worst_feats = None
    worst_seg_idx = -1

    for seg_idx, seg in enumerate(segments):
        feats = extract_50_features(
            seg['主轴'], seg['X'], seg['Y'], seg['force'],
            sampling_rate=sig['sampling_rate'],
        )

        # 加权投票（权重经全量标注数据网格搜索验证）：
        #   规则票为异常(1/2)时是"主动发现"，权重 1.0；
        #   规则票为 0 时只是"未见偏离"（无法区分稳定加工/空载），权重 0.3；
        #   融合模型经过训练验证，是主判据，权重 3.0。
        score = {0: 0.0, 1: 0.0, 2: 0.0}
        for mode in _MODE_LIST:
            mode_result = _run_single_mode(feats, mode, models)
            if mode_result:
                p = mode_result['prediction']
                score[p] += 1.0 if p != 0 else 0.3
        fusion_pred = None
        try:
            fusion_result = diagnose_segment(
                seg['主轴'], seg['X'], seg['Y'], seg['force'],
                sig['sampling_rate'], models,
            )
            fusion_pred = fusion_result['prediction']
            score[fusion_pred] += 3.0
        except Exception:
            pass  # 模型失败时仅依靠规则票，不再强行投"稳定"

        top_score = max(score.values())
        tied = [p for p, s in score.items() if s >= top_score - 1e-9]
        # 平票时优先采纳融合模型结果，其次取严重程度更高的类别
        if fusion_pred is not None and fusion_pred in tied:
            final_pred = fusion_pred
        else:
            final_pred = max(tied)
        all_preds.append(final_pred)

        segment_results.append({
            'idx': seg_idx,
            't_start': seg['t_start'],
            't_end': seg['t_end'],
            'pred': final_pred,
            'cls': CLASS_NAMES[final_pred],
            'feats': feats,
        })

    # ========== 时序平滑（窗口3中值滤波，消除孤立跳变段） ==========
    if len(all_preds) >= 3:
        smoothed = list(all_preds)
        padded = [all_preds[0]] + list(all_preds) + [all_preds[-1]]
        for i in range(len(all_preds)):
            smoothed[i] = int(np.median(padded[i:i + 3]))
        all_preds = smoothed
        for sr, p in zip(segment_results, all_preds):
            sr['pred'] = p
            sr['cls'] = CLASS_NAMES[p]

    # ========== 重点段跟踪（平滑后，颤振优先级最高） ==========
    for sr in segment_results:
        if sr['pred'] > worst_pred:
            worst_pred = sr['pred']
            worst_feats = sr['feats']
            worst_seg_idx = sr['idx']

    # ========== 合并连续同类别段 ==========
    merged = []
    for sr in segment_results:
        if not merged or merged[-1]['pred'] != sr['pred']:
            merged.append({
                'pred': sr['pred'],
                'cls': sr['cls'],
                'start': sr['t_start'],
                'end': sr['t_end'],
            })
        else:
            merged[-1]['end'] = sr['t_end']

    # ========== 重点段特征偏离 ==========
    amp_result = None
    if worst_feats is not None:
        amp_result = _run_single_mode(worst_feats, "chatter_amplitude", models)

    # ========== 生成报告 ==========
    pred_counts = Counter(all_preds)
    dominant_pred = max(pred_counts, key=pred_counts.get)

    report = []
    report.append("━" * 48)
    report.append("  📄 颤振综合诊断报告")
    report.append("━" * 48)

    # 信号概况
    for name, data_key in [('主轴振动', '主轴'), ('X轴振动', 'X'),
                            ('Y轴振动', 'Y'), ('三向力合力', 'force')]:
        d = sig[data_key]
        report.append(f"{name}: 均值={np.mean(np.abs(d)):.4f}  标准差={np.std(d):.4f}  峰值={np.max(np.abs(d)):.4f}")

    report.append("")
    report.append("【诊断概览】")
    report.append(f"总时长: {duration:.0f} s")
    report.append("工况划分:")
    for i, m in enumerate(merged):
        report.append(f"  第{'一二三'[i] if i < 3 else i+1}段{m['start']:.0f}–{m['end']:.0f} s  : {m['cls']}")

    # 重点段分析 (仅颤振段需要重点关注)
    if worst_pred == 2 and worst_seg_idx >= 0:
        ws = segment_results[worst_seg_idx]
        # 时间范围取合并后的颤振整段，与「工况划分」保持一致
        chatter_block = next((m for m in merged
                              if m['pred'] == 2 and m['start'] <= ws['t_start'] <= m['end']), None)
        if chatter_block is None:
            chatter_block = next((m for m in merged if m['pred'] == 2), None)
        block_start = chatter_block['start'] if chatter_block else ws['t_start']
        block_end = chatter_block['end'] if chatter_block else ws['t_end']
        report.append(f"重点段分析：{block_start:.0f}–{block_end:.0f}s {ws['cls']}：关键特征偏离:")
        if amp_result and amp_result.get('key_deviations'):
            for name, fv, mean, std, z in amp_result['key_deviations'][:3]:
                direction = "↑升高" if fv > mean else "↓降低"
                report.append(f"{name}: z={abs(z):.1f} {direction}")
    elif worst_pred == 1:
        report.append("检测到空载运行段，未检测到颤振。")
    else:
        report.append("未检测到明显颤振。")

    report.append("")
    report.append("【工艺建议】")
    if worst_feats is not None:
        report.append(generate_recommendation(worst_pred, worst_feats, models))
    else:
        report.append("设备运行状态正常，无需调整。")

    report.append("")
    report.append("━" * 48)

    return '\n'.join(report)


# ==================== 主入口 (支持多模式) ====================
def diagnose_csv(csv_path: str, long_segment_size: int = SEGMENT_SIZE, mode: str = "chatter_fusion") -> str:
    """
    颤振诊断主函数，支持多种判断模式。

    Args:
        csv_path: CSV文件路径
        long_segment_size: 长信号分段大小(默认256点，必须与训练一致)
        mode: 诊断模式
            chatter_amplitude - 幅值阈值模式
            chatter_frequency - 频谱特征模式
            chatter_timefreq  - 时频分析模式
            chatter_fusion    - 多源融合模式（默认）
            chatter_trend     - 趋势监测模式
            chatter_comprehensive - 综合诊断（全部模式投票）

    Returns:
        诊断报告文本
    """
    # 综合诊断走独立分支
    if mode == "chatter_comprehensive":
        return diagnose_comprehensive(csv_path, long_segment_size=long_segment_size)

    # 实时监控走独立分支
    if mode == "chatter_monitor":
        from .baseline_monitor import monitor_csv
        return monitor_csv(csv_path)

    models = load_models()

    # 读取验证
    result = read_and_validate_csv(csv_path)
    if 'error' in result:
        return f"⚠️ {result['error']}"

    mode_name_map = {
        "chatter_amplitude": "幅值阈值诊断",
        "chatter_frequency": "频谱特征诊断",
        "chatter_timefreq": "时频分析诊断",
        "chatter_fusion": "多源融合诊断",
        "chatter_trend": "趋势监测诊断",
        "chatter_comprehensive": "综合诊断",
    }
    mode_name = mode_name_map.get(mode, "综合诊断")

    sig = result
    report_parts = []

    # 时间戳
    report_parts.append(f"颤振诊断报告 - {mode_name}")
    report_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_parts.append(f"信号文件: {os.path.basename(csv_path)}")
    report_parts.append(f"诊断模式: {mode}")
    report_parts.append("")

    # 信号描述
    report_parts.append("【信号概况】")
    report_parts.append(generate_description(sig))
    report_parts.append("")

    mode_diagnose_func = {
        "chatter_amplitude": diagnose_amplitude_mode,
        "chatter_frequency": diagnose_frequency_mode,
        "chatter_timefreq": diagnose_timefreq_mode,
        "chatter_fusion": diagnose_segment,
        "chatter_trend": diagnose_trend_mode,
    }
    diagnose_func = mode_diagnose_func.get(mode, diagnose_segment)

    # 分段诊断
    segments = segment_signal(
        sig['time'], sig['主轴'], sig['X'], sig['Y'], sig['force'],
        segment_size=long_segment_size,
    )

    if len(segments) == 1:
        if mode != "chatter_fusion":
            feats = extract_50_features(sig['主轴'], sig['X'], sig['Y'], sig['force'], sampling_rate=sig['sampling_rate'])
            seg_result = diagnose_func(feats, models)
        else:
            seg_result = diagnose_segment(
                sig['主轴'], sig['X'], sig['Y'], sig['force'],
                sig['sampling_rate'], models,
            )
        pred = seg_result['prediction']
        conf = seg_result['confidence']

        report_parts.append("【诊断结果】")
        report_parts.append(f"诊断状态: {seg_result['class_name']} (置信度: {conf:.1%})")
        report_parts.append(f"各类别概率: " + ", ".join(
            f"{k}: {v:.1%}" for k, v in seg_result['probabilities'].items()
        ))

        if mode != "chatter_fusion" and 'key_deviations' in seg_result:
            report_parts.append("")
            report_parts.append("【关键偏离特征】")
            for name, fv, mean, std, z in seg_result['key_deviations']:
                direction = "↑升高" if fv > mean else "↓降低"
                report_parts.append(f"  {name}: {fv:.4f} (基线 {mean:.4f}±{std:.4f}) {direction} z={z:.2f}")

        report_parts.append("")
        report_parts.append("【分析依据】")
        report_parts.append(generate_explanation(seg_result['features_50'], models, pred))
        report_parts.append("")

        report_parts.append("【工艺建议】")
        report_parts.append(generate_recommendation(pred, seg_result['features_50'], models))

    else:
        report_parts.append(f"信号分段数: {len(segments)}")
        report_parts.append("")

        chatter_segments = []
        all_preds = []

        for i, seg in enumerate(segments):
            if mode != "chatter_fusion":
                feats = extract_50_features(seg['主轴'], seg['X'], seg['Y'], seg['force'], sampling_rate=sig['sampling_rate'])
                seg_result = diagnose_func(feats, models)
            else:
                seg_result = diagnose_segment(
                    seg['主轴'], seg['X'], seg['Y'], seg['force'],
                    sig['sampling_rate'], models,
                )
            pred = seg_result['prediction']
            all_preds.append(pred)

            seg_label = f"[段{i+1}/{len(segments)}] {seg_result['class_name']} " \
                        f"({seg['t_start']:.1f}s-{seg['t_end']:.1f}s, 置信度{seg_result['confidence']:.1%})"

            if pred == 0:
                report_parts.append(f"  {seg_label}")
            else:
                chatter_segments.append((seg, seg_result, seg_label))
                report_parts.append(f"  {seg_label}")

        from collections import Counter
        pred_counts = Counter(all_preds)
        dominant = max(pred_counts, key=pred_counts.get)

        report_parts.append("")
        report_parts.append(f"【整体诊断】{CLASS_NAMES[dominant]}")
        report_parts.append(f"各段状态分布: " + ", ".join(
            f"{CLASS_NAMES[k]}×{v}" for k, v in sorted(pred_counts.items())
        ))

        if chatter_segments:
            report_parts.append("")
            report_parts.append("【颤振段详细分析】")
            for seg, seg_res, label in chatter_segments:
                report_parts.append(f"\n  {label}")
                if 'key_deviations' in seg_res:
                    for name, fv, mean, std, z in seg_res['key_deviations']:
                        direction = "↑升高" if fv > mean else "↓降低"
                        report_parts.append(f"    {name}: z={z:.2f} {direction}")
                report_parts.append(generate_explanation(seg_res['features_50'], models, seg_res['prediction']))

            report_parts.append("")
            report_parts.append("【工艺建议】")
            report_parts.append(generate_recommendation(
                max(pr['prediction'] for _, pr, _ in chatter_segments),
                chatter_segments[0][1]['features_50'],
                models,
            ))

    return '\n'.join(report_parts)


# 命令行直接运行
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python chatter_diagnosis_skill.py <csv_path>")
        print("示例: python chatter_diagnosis_skill.py signal.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    report = diagnose_csv(csv_file)
    print(report)
