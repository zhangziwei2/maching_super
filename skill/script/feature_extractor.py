"""
固定50维特征提取模块 — v3.0
特征定义严格对齐参考代码 2.2_SAEDimensionreduction.py：

  X轴 15 特征: Clearance Factor, Power Spectrum Clearance, Peak, Peak-to-Peak,
              RMS, Std, STFT Mean, Variance, Signal Energy, Spectral Energy,
              STFT Total Energy, Time-Frequency Entropy, Power Spectrum Peak,
              Frequency Variance, Shape Factor
  Y轴 15 特征: Peak-to-Peak, STFT Mean, Std, RMS, Peak, Power Spectrum Clearance,
              Clearance Factor, Mean Square Frequency, Spectral Centroid,
              Time-Frequency Entropy, Frequency Variance, Skewness, Mean,
              STFT Total Energy, Signal Energy
  Z轴 15 特征: Clearance Factor, Peak-to-Peak, Power Spectrum Clearance, Peak,
              Shape Factor, STFT Mean, Std, RMS, Impulse Factor, Crest Factor,
              Kurtosis, Frequency Variance, Mean Square Frequency, Peak Count,
              Spectral Centroid
  力  5 特征: freq_variance, peak2peak, impulse_factor, peak, crest_factor

共 15*3 + 5 = 50 维。不使用 ReliefF 或任何特征选择。
"""

import numpy as np
from scipy import signal as scipy_signal, stats


# ==================== 公共工具函数 ====================

def _fft_features(data: np.ndarray, fs: float):
    """计算频域特征：频率轴、功率谱、幅值谱"""
    n = len(data)
    if n <= 1:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    fft_vals = np.fft.fft(data)
    freq_mag = np.abs(fft_vals[:n // 2])
    power_spec = freq_mag ** 2
    freq_axis = np.fft.fftfreq(n, 1.0 / fs)[:n // 2]
    return freq_axis, freq_mag, power_spec


def _stft_features(data: np.ndarray, fs: float):
    """计算 STFT 时频特征"""
    n = len(data)
    nperseg = min(256, n)
    noverlap = nperseg // 2
    _, _, Zxx = scipy_signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    stft_mag = np.abs(Zxx)
    return stft_mag


def _spectral_centroid(freq_axis, freq_mag):
    """频谱质心 = sum(f × |X(f)|) / sum(|X(f)|)"""
    total = np.sum(freq_mag)
    if total == 0:
        return 0.0
    return np.sum(freq_axis * freq_mag) / total


def _frequency_variance(freq_axis, freq_mag):
    """频域方差 = sum((f - centroid)² × |X(f)|) / sum(|X(f)|)"""
    total = np.sum(freq_mag)
    if total == 0:
        return 0.0
    cent = _spectral_centroid(freq_axis, freq_mag)
    return np.sum(((freq_axis - cent) ** 2) * freq_mag) / total


def _mean_square_frequency(freq_axis, power_spec):
    """均方频率 = sum(f² × P(f)) / sum(P(f))"""
    total = np.sum(power_spec)
    if total == 0:
        return 0.0
    return np.sum((freq_axis ** 2) * power_spec) / total


def _peak_count(freq_mag, threshold=0.5):
    """频域峰值计数：幅值超过阈值×max 的峰数"""
    if len(freq_mag) < 3:
        return 0
    peak_valley = np.diff(np.sign(np.diff(freq_mag)))
    peaks = np.where(peak_valley < 0)[0] + 1  # 局部极大值索引
    thresh = threshold * np.max(freq_mag) if np.max(freq_mag) > 0 else 0
    return int(np.sum(freq_mag[peaks] > thresh))


def _time_frequency_entropy(stft_mag):
    """时频熵 = -sum(p × log2(p))"""
    total = np.sum(stft_mag)
    if total == 0:
        return 0.0
    norm = stft_mag / total
    return -np.sum(norm * np.log2(norm + 1e-10))


# ==================== X 轴 15 特征 ====================
# 与参考代码 2.2_SAEDimensionreduction.py 的 x_features 严格一致

X_FEATURE_NAMES = [
    'X_Clearance_Factor',        # 0: Time Clearance Factor
    'X_Power_Spectrum_Clearance',# 1: Frequency Power Spectrum Clearance
    'X_Peak',                    # 2: Time Peak
    'X_Peak_to_Peak',            # 3: Time Peak-to-Peak
    'X_RMS',                     # 4: Time RMS
    'X_Std',                     # 5: Time Std
    'X_STFT_Mean',               # 6: STFT Mean
    'X_Variance',                # 7: Time Variance
    'X_Signal_Energy',           # 8: Time Signal Energy
    'X_Spectral_Energy',         # 9: Frequency Spectral Energy
    'X_STFT_Total_Energy',       # 10: STFT Total Energy
    'X_Time_Frequency_Entropy',  # 11: Time-Frequency Entropy
    'X_Power_Spectrum_Peak',     # 12: Frequency Power Spectrum Peak
    'X_Frequency_Variance',      # 13: Frequency Variance
    'X_Shape_Factor',            # 14: Time Shape Factor
]


def extract_x_features(data: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """X轴 15 特征（对齐参考代码 x_features 顺序）"""
    fts = np.zeros(15)
    n = len(data)
    if n == 0:
        return fts

    abs_mean = np.mean(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))

    # 时域
    fts[0] = peak / (abs_mean ** 2) if abs_mean != 0 else 0.0        # Clearance Factor
    fts[2] = peak                                                     # Peak
    fts[3] = np.max(data) - np.min(data)                              # Peak-to-Peak
    fts[4] = rms                                                      # RMS
    fts[5] = np.std(data)                                             # Std
    fts[7] = np.var(data)                                             # Variance
    fts[8] = np.sum(data ** 2)                                        # Signal Energy
    fts[14] = rms / abs_mean if abs_mean != 0 else 0.0               # Shape Factor

    # 频域
    freq_axis, freq_mag, power_spec = _fft_features(data, fs)
    ps_mean = np.mean(power_spec)
    fts[1] = np.max(power_spec) / (ps_mean ** 2) if ps_mean != 0 else 0.0  # Power Spectrum Clearance
    fts[9] = np.sum(freq_mag ** 2)                                           # Spectral Energy
    fts[12] = np.max(power_spec)                                              # Power Spectrum Peak
    fts[13] = _frequency_variance(freq_axis, freq_mag)                       # Frequency Variance

    # 时频
    stft_mag = _stft_features(data, fs)
    fts[6] = np.mean(stft_mag)                                               # STFT Mean
    fts[10] = np.sum(stft_mag ** 2)                                          # STFT Total Energy
    fts[11] = _time_frequency_entropy(stft_mag)                              # Time-Frequency Entropy

    return fts


# ==================== Y 轴 15 特征 ====================

Y_FEATURE_NAMES = [
    'Y_Peak_to_Peak',            # 0: 时域峰峰值
    'Y_STFT_Mean',               # 1: STFT均值
    'Y_Std',                     # 2: 时域标准差
    'Y_RMS',                     # 3: 时域均方根
    'Y_Peak',                    # 4: 时域峰值
    'Y_Power_Spectrum_Clearance',# 5: 频域功率谱间隙
    'Y_Clearance_Factor',        # 6: 时域间隙因子
    'Y_Mean_Square_Frequency',   # 7: 频域均方频率
    'Y_Spectral_Centroid',       # 8: 频域频谱质心
    'Y_Time_Frequency_Entropy',  # 9: 时频熵
    'Y_Frequency_Variance',      # 10: 频域方差
    'Y_Skewness',                # 11: 时域偏度
    'Y_Mean',                    # 12: 时域均值
    'Y_STFT_Total_Energy',       # 13: STFT总能量
    'Y_Signal_Energy',           # 14: 时域信号能量
]


def extract_y_features(data: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Y轴 15 特征"""
    fts = np.zeros(15)
    n = len(data)
    if n == 0:
        return fts

    abs_mean = np.mean(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))

    # 时域
    fts[0] = np.max(data) - np.min(data)                             # Peak-to-Peak
    fts[2] = np.std(data)                                             # Std
    fts[3] = rms                                                      # RMS
    fts[4] = peak                                                     # Peak
    fts[6] = peak / (abs_mean ** 2) if abs_mean != 0 else 0.0        # Clearance Factor
    fts[11] = stats.skew(data)                                        # Skewness
    fts[12] = np.mean(data)                                           # Mean
    fts[14] = np.sum(data ** 2)                                       # Signal Energy

    # 频域
    freq_axis, freq_mag, power_spec = _fft_features(data, fs)
    ps_mean = np.mean(power_spec)
    fts[5] = np.max(power_spec) / (ps_mean ** 2) if ps_mean != 0 else 0.0  # Power Spectrum Clearance
    fts[7] = _mean_square_frequency(freq_axis, power_spec)                 # Mean Square Frequency
    fts[8] = _spectral_centroid(freq_axis, freq_mag)                       # Spectral Centroid
    fts[10] = _frequency_variance(freq_axis, freq_mag)                     # Frequency Variance

    # 时频
    stft_mag = _stft_features(data, fs)
    fts[1] = np.mean(stft_mag)                                               # STFT Mean
    fts[9] = _time_frequency_entropy(stft_mag)                              # Time-Frequency Entropy
    fts[13] = np.sum(stft_mag ** 2)                                          # STFT Total Energy

    return fts


# ==================== Z 轴 15 特征 ====================
# 与参考代码 2.2_SAEDimensionreduction.py 的 z_features 严格一致

Z_FEATURE_NAMES = [
    'Z_Clearance_Factor',        # 0: Time Clearance Factor
    'Z_Peak_to_Peak',            # 1: Time Peak-to-Peak
    'Z_Power_Spectrum_Clearance',# 2: Frequency Power Spectrum Clearance
    'Z_Peak',                    # 3: Time Peak
    'Z_Shape_Factor',            # 4: Time Shape Factor
    'Z_STFT_Mean',               # 5: STFT Mean
    'Z_Std',                     # 6: Time Std
    'Z_RMS',                     # 7: Time RMS
    'Z_Impulse_Factor',          # 8: Time Impulse Factor
    'Z_Crest_Factor',            # 9: Time Crest Factor
    'Z_Kurtosis',                # 10: Time Kurtosis
    'Z_Frequency_Variance',      # 11: Frequency Variance
    'Z_Mean_Square_Frequency',   # 12: Frequency Mean Square Frequency
    'Z_Peak_Count',              # 13: Frequency Peak Count
    'Z_Spectral_Centroid',       # 14: Frequency Spectral Centroid
]


def extract_z_features(data: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Z轴 15 特征（对齐参考代码 z_features 顺序）"""
    fts = np.zeros(15)
    n = len(data)
    if n == 0:
        return fts

    abs_mean = np.mean(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))

    # 时域
    fts[0] = peak / (abs_mean ** 2) if abs_mean != 0 else 0.0        # Clearance Factor
    fts[1] = np.max(data) - np.min(data)                             # Peak-to-Peak
    fts[3] = peak                                                     # Peak
    fts[4] = rms / abs_mean if abs_mean != 0 else 0.0                # Shape Factor
    fts[6] = np.std(data)                                             # Std
    fts[7] = rms                                                      # RMS
    fts[8] = peak / abs_mean if abs_mean != 0 else 0.0               # Impulse Factor
    fts[9] = peak / rms if rms != 0 else 0.0                         # Crest Factor
    fts[10] = stats.kurtosis(data, fisher=False)                     # Kurtosis

    # 频域
    freq_axis, freq_mag, power_spec = _fft_features(data, fs)
    ps_mean = np.mean(power_spec)
    fts[2] = np.max(power_spec) / (ps_mean ** 2) if ps_mean != 0 else 0.0  # Power Spectrum Clearance
    fts[11] = _frequency_variance(freq_axis, freq_mag)                      # Frequency Variance
    fts[12] = _mean_square_frequency(freq_axis, power_spec)                 # Mean Square Frequency
    fts[13] = _peak_count(freq_mag)                                          # Peak Count
    fts[14] = _spectral_centroid(freq_axis, freq_mag)                       # Spectral Centroid

    # 时频
    stft_mag = _stft_features(data, fs)
    fts[5] = np.mean(stft_mag)                                               # STFT Mean

    return fts


# ==================== 力 5 特征 ====================

FORCE_FEATURE_NAMES = [
    'Force_Freq_Variance',        # 0: 合力频域方差
    'Force_Peak2Peak',            # 1: 合力时域峰峰值
    'Force_Impulse_Factor',       # 2: 合力时域脉冲因子
    'Force_Peak',                 # 3: 合力时域峰值
    'Force_Crest_Factor',         # 4: 合力时域峰值因子
]


def extract_force_features(data: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """力 5 特征"""
    fts = np.zeros(5)
    n = len(data)
    if n == 0:
        return fts

    peak = np.max(np.abs(data))
    abs_mean = np.mean(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))

    # 时域
    fts[1] = np.max(data) - np.min(data)                             # Peak-to-Peak
    fts[2] = peak / abs_mean if abs_mean != 0 else 0.0               # Impulse Factor
    fts[3] = peak                                                     # Peak
    fts[4] = peak / rms if rms != 0 else 0.0                         # Crest Factor

    # 频域
    freq_axis, freq_mag, _ = _fft_features(data, fs)
    fts[0] = _frequency_variance(freq_axis, freq_mag)                 # Frequency Variance

    return fts


# ==================== 组合 ====================

ALL_FEATURE_NAMES = (
    X_FEATURE_NAMES + Y_FEATURE_NAMES + Z_FEATURE_NAMES + FORCE_FEATURE_NAMES
)
"""全部 50 个特征名，用于训练时列名"""


def extract_50_features(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    z_signal: np.ndarray,
    force_signal: np.ndarray,
    sampling_rate: float = 1000.0,
) -> np.ndarray:
    """
    从 X/Y/Z 振动 + 力 信号中提取 50 维特征。

    Args:
        x_signal: X轴振动信号 (1D)
        y_signal: Y轴振动信号 (1D)
        z_signal: Z轴振动信号 (1D)
        force_signal: 三向力合力信号 (1D)
        sampling_rate: 采样率 (Hz)

    Returns:
        50维特征向量
    """
    x_feat = extract_x_features(x_signal, sampling_rate)       # 15
    y_feat = extract_y_features(y_signal, sampling_rate)       # 15
    z_feat = extract_z_features(z_signal, sampling_rate)       # 15
    force_feat = extract_force_features(force_signal, sampling_rate)  # 5
    return np.concatenate([x_feat, y_feat, z_feat, force_feat])
