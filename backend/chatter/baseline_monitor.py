"""
设备专属基线管理 + 实时监控报警模块

功能:
  1. compute_baseline(csv_path)  — 上传稳定切削 CSV，建立设备专属50维特征基线
  2. monitor_csv(csv_path)       — 上传待检测 CSV，与基线比对并输出报警报告
  3. baseline_status()           — 查询基线注册状态

依赖:
  - read_and_validate_csv, segment_signal (chatter_diagnosis_skill.py)
  - extract_50_features, VIB_FEATURE_NAMES, FORCE_FEATURE_NAMES (feature_extractor.py)
"""

import os
import json
import numpy as np
from datetime import datetime

from .feature_extractor import extract_50_features, VIB_FEATURE_NAMES, FORCE_FEATURE_NAMES

# ==================== 路径与常量 ====================

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_MODULE_DIR, 'models')
_USER_BASELINE_PATH = os.path.join(_MODEL_DIR, 'user_baseline.json')

# 所有50个特征名（与 feature_extractor 保持完全一致）
ALL_FEATURE_NAMES = (
    [f"Vib{i}_{n}" for i in range(3) for n in VIB_FEATURE_NAMES]
    + [f"Force_{n}" for n in FORCE_FEATURE_NAMES]
)

# 基线数据格式版本。v3 起：log 域统计 + 自适应阈值标定
BASELINE_VERSION = 3

# z-score 阈值（仅在基线缺少标定信息时作为兜底）
Z_ALERT = 2.0   # 关注线（旧版兼容用）
Z_ALARM = 3.5   # 报警线（旧版兼容用）

# 段级偏离分默认阈值（v3 无标定时兜底）
DEFAULT_SCORE_ALERT = 3.5
DEFAULT_SCORE_ALARM = 5.0

# 段级评分：取偏离最大的前 K 个特征求均值，避免单一特征极值主导
SCORE_TOPK = 5

# 尺度下限系数：防止某特征在基线段近似恒定时 std→0 导致 z 爆表
STD_FLOOR_RATIO = 0.02
STD_FLOOR_ABS = 1e-3

# 全局结论按「异常段占比」分级，避免单段异常即判全局停机
RATIO_ALARM = 0.30   # 异常段占比 ≥30% → 报警
RATIO_WARN = 0.10    # 异常段占比 ≥10% → 关注

# 状态图标
STATUS_ICON = {0: "\U0001f7e2", 1: "\U0001f7e1", 2: "\U0001f534"}  # 🟢 🟡 🔴
STATUS_LABEL = {0: "正常", 1: "关注", 2: "报警"}
STATUS_EMOJI = {0: "✅", 1: "⚠️", 2: "\U0001f534"}  # ✅ ⚠️ 🔴


# ==================== 工具函数 ====================

def _log_transform(x):
    """
    符号保持的对数压缩。

    50 维特征里能量/方差类量纲跨几个数量级且呈重尾分布，
    直接算 z-score 会让这些特征的正常波动也轻易突破阈值。
    取 log 后分布接近正态，z-score 才有统计意义。
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _apply_std_floor(std, mean):
    """给标准差加下限，防止基线段内近似恒定的特征把 z 放大到失真。"""
    std = np.asarray(std, dtype=float)
    mean = np.asarray(mean, dtype=float)
    floor = np.maximum(np.abs(mean) * STD_FLOOR_RATIO, STD_FLOOR_ABS)
    return np.maximum(std, floor)


def _segment_score(z_values) -> float:
    """
    段级偏离分 = 偏离最大的前 K 个特征 z 的均值。

    原实现取 50 维的 max(z)，属极值统计：特征越多越容易偶然爆表，
    健康段也会被判异常。取 top-K 均值需要多个特征同时偏离才升高，
    显著抑制误报。
    """
    z = np.asarray(z_values, dtype=float)
    if z.size == 0:
        return 0.0
    k = min(SCORE_TOPK, z.size)
    return float(np.sort(z)[-k:].mean())


def _status_from_score(score: float, alert: float, alarm: float) -> int:
    """根据段级偏离分与阈值返回状态: 0=正常, 1=关注, 2=报警"""
    if score >= alarm:
        return 2
    if score >= alert:
        return 1
    return 0


def _compute_segment_status(max_z: float) -> int:
    """根据最大 z 值返回段状态: 0=正常, 1=关注, 2=报警（旧版兼容保留）"""
    if max_z >= Z_ALARM:
        return 2
    elif max_z >= Z_ALERT:
        return 1
    return 0


def _get_direction(value: float, mean: float) -> str:
    """返回偏离方向箭头"""
    return "↑升高" if value > mean else "↓降低"  # ↑升高 / ↓降低


# ==================== 核心函数 ====================

def _plot_signal_for(csv_path, sig):
    """为给定 CSV 生成 4×1 信号图（v3.1），返回保存路径或 None。

    实现复用 chatter_diagnosis_skill._plot_signal_for，避免两处维护同一逻辑。
    绘图失败只返回 None，不抛异常——信号图是附属产物，不应阻断监控流程。
    """
    try:
        from .chatter_diagnosis_skill import _plot_signal_for as _impl
        return _impl(csv_path, sig)
    except Exception as exc:
        print(f"[baseline_monitor] 信号图生成失败（不影响监控）: {exc}")
        return None


def baseline_status() -> dict:
    """
    查询基线状态。

    Returns:
        dict: {
            "exists": bool,
            "meta": { "created_at": str, "source_file": str, "num_segments": int, "total_samples": int } | None
        }
    """
    if not os.path.exists(_USER_BASELINE_PATH):
        return {"exists": False, "meta": None}
    try:
        with open(_USER_BASELINE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "meta": data.get("meta", {})}
    except (json.JSONDecodeError, KeyError):
        return {"exists": False, "meta": None}


def _read_conditions(csv_path: str) -> list:
    """
    读取 CSV 第6列（工况标签），返回每行的工况字符串列表。
    如果列不存在或读取失败，返回空列表。
    """
    from .chatter_diagnosis_skill import _read_csv_with_encoding
    try:
        df = _read_csv_with_encoding(csv_path, header=None, skiprows=1)
        if df.shape[1] >= 6:
            conditions = df.iloc[:, 5].astype(str).str.strip().values
            return conditions.tolist()
    except Exception:
        pass
    return []


def _segment_conditions(conditions: list, segment_size: int = 256, n_samples: int = 0) -> list:
    """
    将逐行工况标签按照 256 点分段聚合，每段取众数作为该段工况。
    Returns: [(segment_idx, dominant_condition), ...]
    """
    if not conditions:
        return []
    from collections import Counter
    seg_conditions = []
    starts = list(range(0, len(conditions) - segment_size + 1, segment_size))
    last_start = (len(conditions) // segment_size) * segment_size
    valid_ranges = starts + ([last_start] if last_start < len(conditions) and (len(conditions) - last_start) >= 256 else [])
    for start in valid_ranges:
        end = min(start + segment_size, len(conditions))
        chunk = conditions[start:end]
        counter = Counter(chunk)
        dominant = counter.most_common(1)[0][0]
        seg_conditions.append(dominant)
    return seg_conditions


def compute_baseline(csv_path: str, condition_filter: str = "") -> dict:
    """
    从 CSV 文件建立设备专属基线，可按工况标签筛选段。

    Args:
        csv_path: CSV 文件路径
        condition_filter: 工况筛选条件，如 "正常加工"、"空载"。
                         仅使用匹配该标签的段计算基线。
                         为空时使用所有段。

    Returns:
        dict: { "status": "ok"|"error", "message": str, ... }
    """
    from .chatter_diagnosis_skill import read_and_validate_csv, segment_signal

    # 读取并验证 CSV
    result = read_and_validate_csv(csv_path)
    if "error" in result:
        return {"status": "error", "message": result["error"]}

    sig = result
    segments = segment_signal(
        sig["time"], sig["主轴"], sig["X"], sig["Y"], sig["force"],
        segment_size=256,
    )

    if len(segments) < 1:
        return {"status": "error", "message": "信号分段数为0，无法建立基线"}

    # 读取工况标签
    raw_conditions = _read_conditions(csv_path)
    seg_conditions = _segment_conditions(raw_conditions, segment_size=256, n_samples=sig["n_samples"])

    # 筛选段
    selected_indices = list(range(len(segments)))
    filter_desc = "全部段"
    if condition_filter and seg_conditions:
        selected_indices = [
            i for i in range(len(segments))
            if i < len(seg_conditions) and seg_conditions[i] == condition_filter
        ]
        filter_desc = f"仅「{condition_filter}」段"

    if not selected_indices:
        return {
            "status": "error",
            "message": f"筛选条件「{condition_filter}」未匹配到任何段。"
                       f"CSV中的工况标签: {set(seg_conditions) if seg_conditions else '未找到'}",
        }

    # 提取选中段的50维特征
    all_features = []
    for idx in selected_indices:
        seg = segments[idx]
        feats = extract_50_features(
            seg["主轴"], seg["X"], seg["Y"], seg["force"],
            sampling_rate=sig["sampling_rate"],
        )
        all_features.append(feats)

    all_features = np.array(all_features)

    # ---- log 域统计（真正用于 z-score 判定）----
    log_features = _log_transform(all_features)
    log_mean = log_features.mean(axis=0)
    log_std = _apply_std_floor(log_features.std(axis=0), log_mean)

    # ---- 原始域统计（仅用于报告里展示"基线 x±y, 当前 z"）----
    raw_mean = all_features.mean(axis=0)
    raw_std = all_features.std(axis=0)

    features_dict = {}
    for i, name in enumerate(ALL_FEATURE_NAMES):
        features_dict[name] = {
            # mean/std 保持原始域，兼容旧版读取逻辑与报告展示
            "mean": float(raw_mean[i]),
            "std": float(raw_std[i]) if raw_std[i] > 1e-12 else 1e-6,
            # log 域统计，v3 判定实际使用
            "log_mean": float(log_mean[i]),
            "log_std": float(log_std[i]),
        }

    # ---- 自适应阈值标定 ----
    # 用基线段自身回算偏离分，取其 p95/p99 作为该设备的关注线/报警线。
    # 固定阈值（2.0/3.5）无视设备与工况的固有波动，是健康段被误报的直接原因。
    z_base = np.abs(log_features - log_mean) / log_std
    base_scores = np.array([_segment_score(row) for row in z_base])
    if base_scores.size >= 5:
        cal_alert = float(np.percentile(base_scores, 95))
        cal_alarm = float(np.percentile(base_scores, 99))
    else:
        # 段数太少，分位数不可靠，退回默认阈值
        cal_alert, cal_alarm = DEFAULT_SCORE_ALERT, DEFAULT_SCORE_ALARM
    # 阈值不低于兜底值，防止基线段过度一致导致阈值贴地
    cal_alert = max(cal_alert, DEFAULT_SCORE_ALERT)
    cal_alarm = max(cal_alarm, cal_alert * 1.3, DEFAULT_SCORE_ALARM)

    calibration = {
        "score_alert": cal_alert,
        "score_alarm": cal_alarm,
        "base_score_p50": float(np.percentile(base_scores, 50)) if base_scores.size else 0.0,
        "base_score_p95": float(np.percentile(base_scores, 95)) if base_scores.size else 0.0,
        "base_score_max": float(base_scores.max()) if base_scores.size else 0.0,
        "n_calib_segments": int(base_scores.size),
        "topk": SCORE_TOPK,
    }

    # 构建基线数据
    baseline_data = {
        "version": BASELINE_VERSION,
        "meta": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": os.path.basename(csv_path),
            "filter": filter_desc,
            "condition_filter": condition_filter,
            "total_segments": len(segments),
            "used_segments": len(selected_indices),
            "total_samples": sig["n_samples"],
            "seg_conditions": seg_conditions,  # 各段工况标签，供参考
        },
        "calibration": calibration,
        "features": features_dict,
    }

    # 保存
    os.makedirs(_MODEL_DIR, exist_ok=True)
    with open(_USER_BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "message": (
            f"基线建立成功: 用 {len(selected_indices)}/{len(segments)} 段, "
            f"{sig['n_samples']} 采样点（{filter_desc}）"
        ),
        "num_segments": len(segments),
        "used_segments": len(selected_indices),
        "total_segments": len(segments),
        "total_samples": sig["n_samples"],
        "meta": baseline_data["meta"],
        "calibration": calibration,
    }


def _load_baseline_arrays(baseline_data: dict):
    """
    从基线 JSON 还原成判定用的 numpy 数组。

    返回: (log_mean, log_std, raw_mean, raw_std, valid_mask, is_v3)
    v3 基线直接读 log_mean/log_std；旧版基线没有 log 域统计，
    退化为用原始域 mean 现算 log 中心，std 用相对波动近似，保证不报错。
    """
    feats = baseline_data.get("features", {})
    is_v3 = int(baseline_data.get("version", 1)) >= 3

    n = len(ALL_FEATURE_NAMES)
    log_mean = np.zeros(n)
    log_std = np.ones(n)
    raw_mean = np.zeros(n)
    raw_std = np.ones(n)
    valid = np.zeros(n, dtype=bool)

    for i, name in enumerate(ALL_FEATURE_NAMES):
        item = feats.get(name)
        if not item:
            continue
        valid[i] = True
        raw_mean[i] = float(item.get("mean", 0.0))
        raw_std[i] = float(item.get("std", 1e-6)) or 1e-6
        if is_v3 and "log_mean" in item:
            log_mean[i] = float(item["log_mean"])
            log_std[i] = float(item.get("log_std", 1.0)) or 1.0
        else:
            # 旧基线兼容：由原始域 mean±std 推算 log 域中心与尺度
            lm = float(np.sign(raw_mean[i]) * np.log1p(abs(raw_mean[i])))
            hi = float(np.sign(raw_mean[i]) * np.log1p(abs(raw_mean[i]) + raw_std[i]))
            log_mean[i] = lm
            log_std[i] = max(abs(hi - lm), abs(lm) * STD_FLOOR_RATIO, STD_FLOOR_ABS)

    log_std = np.maximum(log_std, STD_FLOOR_ABS)
    return log_mean, log_std, raw_mean, raw_std, valid, is_v3


def monitor_csv(csv_path: str) -> str:
    """
    监控诊断：双轨判定并输出报告。

      轨一（主判据）出厂融合模型对每段做绝对分类：稳定加工 / 空载 / 颤振。
      轨二（辅判据）与用户基线比对，给出相对偏离分，用于早期劣化预警。

    颤振与否由轨一决定；轨二只负责"相对本机健康态是否在劣化"，
    不会单独把健康段判成颤振。

    Args:
        csv_path: 待检测 CSV 文件路径

    Returns:
        str: 结构化的监控报警报告文本

    Raises:
        FileNotFoundError: 基线未建立
    """
    if not os.path.exists(_USER_BASELINE_PATH):
        raise FileNotFoundError(
            "设备基线尚未建立。请先上传一段稳定切削的 CSV 文件注册基线:\n"
            "  POST /baseline/register"
        )

    with open(_USER_BASELINE_PATH, encoding="utf-8") as f:
        baseline_data = json.load(f)
    meta = baseline_data.get("meta", {})
    calib = baseline_data.get("calibration", {})

    log_mean, log_std, raw_mean, raw_std, valid_mask, is_v3 = _load_baseline_arrays(baseline_data)
    if not valid_mask.any():
        return "⚠️ 基线数据损坏（无有效特征），请重新注册基线。"

    score_alert = float(calib.get("score_alert", DEFAULT_SCORE_ALERT))
    score_alarm = float(calib.get("score_alarm", DEFAULT_SCORE_ALARM))

    from .chatter_diagnosis_skill import read_and_validate_csv, segment_signal

    result = read_and_validate_csv(csv_path)
    if "error" in result:
        return f"⚠️ {result['error']}"

    # 【v3.1】读入即绘制 4×1 原始信号图（先于监控报告生成）
    _plot_signal_for(csv_path, result)

    sig = result
    segments = segment_signal(
        sig["time"], sig["主轴"], sig["X"], sig["Y"], sig["force"],
        segment_size=256,
    )
    if not segments:
        return "⚠️ 信号分段数为0，无法监测。"

    # ---- 轨一：加载出厂融合模型（失败则降级为纯基线模式）----
    models = None
    model_error = ""
    try:
        from .chatter_diagnosis_skill import load_models, diagnose_segment, CLASS_NAMES
        models = load_models(silent=True)
    except Exception as e:  # 模型缺失/损坏不应导致监测整体失败
        model_error = str(e)
        CLASS_NAMES = ["稳定加工", "空载", "颤振"]

    report_parts = []
    report_parts.append("━" * 48)
    report_parts.append("  \U0001f4a1 实时监控报警报告")
    report_parts.append("━" * 48)
    report_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_parts.append(f"信号文件: {os.path.basename(csv_path)}")
    report_parts.append(f"基线文件: {meta.get('source_file', 'unknown')}")
    report_parts.append(f"基线工况: {meta.get('filter', '未知')}")
    report_parts.append(f"基线建立: {meta.get('created_at', 'unknown')}")
    if is_v3:
        report_parts.append(
            f"判定阈值: 关注≥{score_alert:.2f} / 报警≥{score_alarm:.2f}"
            f"（由基线自身波动标定）"
        )
    else:
        report_parts.append("判定阈值: 兜底阈值（旧版基线，建议重新注册以启用自适应阈值）")
    if models is None:
        report_parts.append(f"⚠️ 出厂模型不可用，已降级为纯基线偏离模式：{model_error[:60]}")
    report_parts.append("")

    # ========== 逐段双轨判定 ==========
    segment_results = []
    all_scores = []
    cls_counter = {name: 0 for name in CLASS_NAMES}

    for seg_idx, seg in enumerate(segments):
        feats = extract_50_features(
            seg["主轴"], seg["X"], seg["Y"], seg["force"],
            sampling_rate=sig["sampling_rate"],
        )

        # --- 轨二：基线相对偏离 ---
        log_feats = _log_transform(feats)
        z_all = np.abs(log_feats - log_mean) / log_std
        z_all = np.where(valid_mask, z_all, 0.0)
        score = _segment_score(z_all[valid_mask])
        all_scores.append(score)

        deviations = []
        for i in np.argsort(z_all)[::-1][:5]:
            if not valid_mask[i] or z_all[i] < 1.0:
                continue
            deviations.append((float(z_all[i]), ALL_FEATURE_NAMES[i],
                               float(feats[i]), float(raw_mean[i]), float(raw_std[i])))

        dev_status = _status_from_score(score, score_alert, score_alarm)

        # --- 轨一：出厂模型绝对分类 ---
        cls_name, conf = "-", 0.0
        if models is not None:
            try:
                r = diagnose_segment(
                    seg["主轴"], seg["X"], seg["Y"], seg["force"],
                    sig["sampling_rate"], models,
                )
                cls_name = r["class_name"]
                conf = float(r["confidence"])
                cls_counter[cls_name] = cls_counter.get(cls_name, 0) + 1
            except Exception:
                cls_name, conf = "-", 0.0

        # --- 融合判定 ---
        # 颤振只由模型认定；基线偏离最高只升到"关注"，避免健康段被误报颤振
        if cls_name == "颤振":
            status = 2
        elif cls_name == "空载":
            status = 1
        elif models is None:
            status = dev_status               # 纯基线模式
        else:
            status = 1 if dev_status >= 2 else 0   # 模型判稳定：偏离超报警线仅作劣化关注

        segment_results.append({
            "idx": seg_idx,
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "status": status,
            "score": score,
            "dev_status": dev_status,
            "cls": cls_name,
            "conf": conf,
            "deviations": deviations,
        })

    total = len(segment_results)

    # ========== 监控概览表 ==========
    report_parts.append("\U0001f6a6 监控概览:")
    header = f"  {'段':<5} {'时间段':<14} {'模型判定':<10} {'置信':<7} {'偏离分':<8} 状态"
    report_parts.append(header)
    report_parts.append("  " + "-" * 62)
    MAX_ROWS = 30
    for sr in segment_results[:MAX_ROWS]:
        icon = STATUS_ICON[sr["status"]]
        label = STATUS_LABEL[sr["status"]]
        conf_txt = f"{sr['conf']*100:.0f}%" if sr["conf"] else "-"
        report_parts.append(
            f"  {sr['idx']+1:<5} {sr['t_start']:.1f}s-{sr['t_end']:.1f}s  "
            f"{sr['cls']:<10} {conf_txt:<7} {sr['score']:<8.2f} {icon}{label}"
        )
    if total > MAX_ROWS:
        report_parts.append(f"  ... 共 {total} 段，仅展示前 {MAX_ROWS} 段")
    report_parts.append("")

    # ========== 分类统计 ==========
    if models is not None:
        report_parts.append("\U0001f4ca 出厂模型判定分布:")
        for name in CLASS_NAMES:
            c = cls_counter.get(name, 0)
            report_parts.append(f"  {name:<8} {c:>4} 段 ({c/total*100:.1f}%)")
        report_parts.append("")

    # ========== 报警段详情 ==========
    alarm_segments = [sr for sr in segment_results if sr["status"] == 2]
    warn_segments = [sr for sr in segment_results if sr["status"] == 1]

    if alarm_segments:
        report_parts.append(f"\U0001f534 报警段详情 ({len(alarm_segments)}段):")
        for sr in alarm_segments[:3]:
            report_parts.append(
                f"  段{sr['idx']+1} ({sr['t_start']:.1f}s-{sr['t_end']:.1f}s) "
                f"模型判定={sr['cls']}(置信{sr['conf']*100:.0f}%) 偏离分={sr['score']:.2f}"
            )
            for z, name, fv, mean_v, std_v in sr["deviations"][:3]:
                direction = _get_direction(fv, mean_v)
                report_parts.append(
                    f"    {name}: z={z:.1f} {direction} (基线 {mean_v:.4f}±{std_v:.4f}, 当前 {fv:.4f})"
                )
        if len(alarm_segments) > 3:
            report_parts.append(f"    ... 还有 {len(alarm_segments)-3} 段报警")
        report_parts.append("")

    if warn_segments:
        report_parts.append(f"⚠️ 关注段 ({len(warn_segments)}段):")
        for sr in warn_segments[:3]:
            reason = sr["cls"] if sr["cls"] in ("空载", "颤振") else f"偏离基线 {sr['score']:.2f}"
            report_parts.append(f"  段{sr['idx']+1}: {reason}")
        report_parts.append("")

    # ========== 趋势分析 ==========
    if len(all_scores) >= 2:
        report_parts.append("\U0001f4c8 趋势分析:")
        half = len(all_scores) // 2
        first_half = float(np.mean(all_scores[:half])) if half else 0.0
        second_half = float(np.mean(all_scores[half:]))
        if second_half > first_half * 1.5:
            trend = "\U0001f4c8 偏离程度持续升高，情况劣化"
        elif second_half < first_half * 0.5:
            trend = "\U0001f4c9 偏离程度下降，状态好转"
        else:
            trend = "↔️ 偏离水平趋于稳定"
        report_parts.append(
            f"  偏离分变化: {all_scores[0]:.2f} → {all_scores[-1]:.2f} "
            f"(前半均值 {first_half:.2f}, 后半均值 {second_half:.2f})"
        )
        report_parts.append(f"  {trend}")
        report_parts.append("")

    # ========== 综合结论（按占比分级，不再一段异常即全局停机）==========
    alarm_count = len(alarm_segments)
    warn_count = len(warn_segments)
    ok_count = total - alarm_count - warn_count
    chatter_count = cls_counter.get("颤振", 0)
    idle_count = cls_counter.get("空载", 0)
    chatter_ratio = chatter_count / total if total else 0.0
    idle_ratio = idle_count / total if total else 0.0
    dev_warn_ratio = sum(1 for sr in segment_results if sr["dev_status"] >= 2) / total if total else 0.0

    if models is not None and chatter_ratio >= RATIO_ALARM:
        conclusion = (
            f"\U0001f534 综合结论: 检出颤振 {chatter_count}/{total} 段"
            f"（{chatter_ratio*100:.0f}%），建议立即降速/调整切削参数并停机检查"
        )
    elif models is not None and chatter_ratio >= RATIO_WARN:
        conclusion = (
            f"⚠️ 综合结论: 间歇性颤振 {chatter_count}/{total} 段"
            f"（{chatter_ratio*100:.0f}%），建议观察并微调主轴转速"
        )
    elif models is not None and idle_ratio >= RATIO_ALARM:
        conclusion = (
            f"⚠️ 综合结论: {idle_count}/{total} 段为空载"
            f"（{idle_ratio*100:.0f}%），刀具可能未接触工件，请核对加工程序"
        )
    elif dev_warn_ratio >= RATIO_ALARM:
        conclusion = (
            f"⚠️ 综合结论: 未检出颤振，但 {dev_warn_ratio*100:.0f}% 的段显著偏离设备基线，"
            f"提示设备状态漂移，建议检查刀具磨损与装夹"
        )
    else:
        conclusion = (
            f"✅ 综合结论: {ok_count}/{total} 段正常，设备运行平稳"
        )
    report_parts.append(conclusion)
    report_parts.append(
        f"  加工状态分布: {STATUS_EMOJI[2]}{alarm_count}段 "
        f"{STATUS_EMOJI[1]}{warn_count}段 {STATUS_EMOJI[0]}{ok_count}段"
    )

    report_parts.append("")
    report_parts.append("━" * 48)

    return "\n".join(report_parts)
