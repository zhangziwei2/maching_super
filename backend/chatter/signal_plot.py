"""
信号可视化模块 — 绘制 4×1 原始信号时序图（v3.1 新增，从 skill/script 合入）

布局（自上而下，与业务语义一致）:
  ① 主轴振动   —— CSV 第 2 列（轴向/主轴方向振动）
  ② X 轴振动   —— CSV 第 3 列
  ③ Y 轴振动   —— CSV 第 4 列
  ④ 三向力合力 —— CSV 第 5 列

列序以 chatter_diagnosis_skill.read_and_validate_csv() 的读取顺序为准
（时间 / 主轴 / X / Y / 力），本模块只消费其返回的语义键名。

四个子图共享 X 轴（时间, 单位 s），便于纵向对齐观察各通道的相位/幅值关系。

设计原则:
  - 不依赖任何诊断模型（SAE / Scaler / 分类器），可在 read_and_validate_csv
    返回后立即调用，即"生成诊断报告前"先给出原始信号概览。
  - 使用 Agg 后端，确保在无显示器的沙箱/服务器环境中也能稳定出图。
  - 对超大信号做自动降采样，避免 PNG 体积膨胀与绘制卡顿。

⚠️ 本模块唯一的非标准库依赖是 matplotlib，且绘图失败一律返回 None，
   不抛异常——信号图是诊断报告的附属产物，不应阻断诊断主流程。
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')  # 无界面/沙箱安全：必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==================== 中文字体配置 ====================
# DejaVu Sans 等默认字体不含 CJK 字形，中文标题/标签会渲染成"豆腐块"。
# 此处按优先级探测系统中可用的中文字体，确保图面中文正确显示。
_CJK_FONT_CANDIDATES = [
    'Microsoft YaHei', '微软雅黑',
    'SimHei', '黑体',
    'SimSun', '宋体',
    'Noto Sans CJK SC', 'Source Han Sans SC',
    'WenQuanYi Micro Hei',
]


def _setup_cjk_font():
    """配置中文字体，避免中文标签乱码；找不到则退回默认字体（中文显示为方框）。

    Returns:
        选中的字体名；无可用中文字体时返回 None。
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in _CJK_FONT_CANDIDATES:
        if cand in available:
            plt.rcParams['font.sans-serif'] = [cand, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
            return cand
    # 环境无中文字体：保留默认，仅提示一次
    print("[signal_plot] 未检测到中文字体，图中中文将显示为方框；"
          "可在服务器安装 Microsoft YaHei / Noto Sans CJK 后重试。")
    return None


_setup_cjk_font()

# ==================== 子图层定义 ====================
# (信号键, 标题, Y 轴标签) —— 顺序即自上而下绘制顺序
#
# ⚠️ 键名必须匹配 chatter_diagnosis_skill.read_and_validate_csv() 的返回：
#    {'time', '主轴', 'X', 'Y', 'force', ...}
# 合入时曾沿用 skill/script 版的 ('z','x','y','force')，与本仓库的键名不一致，
# 导致 sig.get(key) 全为 None、四张子图被 set_visible(False) 静默留白。
_SIGNAL_LAYERS = [
    ('主轴',   '主轴振动',   '振动幅值'),
    ('X',      'X 轴振动',   '振动幅值'),
    ('Y',      'Y 轴振动',   '振动幅值'),
    ('force',  '三向力合力', '力合力 (N)'),
]

# 统一主色：克制、专业，避免花哨
_ACCENT = '#2c7be5'
# 主轴层稍微加深，突出"主轴"这一关键通道
_SPINDLE_ACCENT = '#e6532c'

# 超过该点数时自动降采样，控制绘制开销与文件体积
_MAX_PLOT_POINTS = 20000


def _maybe_downsample(time, data):
    """信号过长时按整数步长抽稀，保留整体形态。"""
    n = len(time)
    if n <= _MAX_PLOT_POINTS:
        return time, data
    stride = int(np.ceil(n / _MAX_PLOT_POINTS))
    return time[::stride], data[::stride]


def plot_signals(sig, out_path=None, dpi=120, show=False):
    """绘制 4×1 原始信号时序图并保存为 PNG。

    Args:
        sig:      read_and_validate_csv 返回的字典，需含
                  time / x / y / z / force 字段。
        out_path: 输出 PNG 路径（str 或 Path）。为 None 时保存到
                  当前工作目录下的 `signal_plot.png`。
        dpi:      图像分辨率，默认 120。
        show:     是否尝试弹出交互窗口（无显示器环境下无效，仅调试用）。

    Returns:
        成功返回保存的文件路径 (str)；输入无效或绘制失败返回 None。
    """
    time = sig.get('time')
    if time is None or len(time) == 0:
        return None

    try:
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        fig.suptitle('颤振诊断 — 原始信号概览', fontsize=14, fontweight='bold')

        for ax, (key, title, ylabel) in zip(axes, _SIGNAL_LAYERS):
            data = sig.get(key)
            if data is None:
                ax.set_visible(False)
                continue

            t_plot, d_plot = _maybe_downsample(time, np.asarray(data, dtype=float))
            color = _SPINDLE_ACCENT if key == 'z' else _ACCENT
            ax.plot(t_plot, d_plot, color=color, linewidth=0.8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, loc='left', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.margins(x=0)
            ax.tick_params(axis='both', labelsize=9)

        axes[-1].set_xlabel('时间 (s)', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if out_path is None:
            out_path = Path.cwd() / 'signal_plot.png'
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        if show:
            # 仅在有显示器环境生效；沙箱环境静默忽略
            try:
                plt.show()
            except Exception:
                pass

        return str(out_path)
    except Exception as exc:
        print(f"[signal_plot] 绘图失败: {exc}")
        return None


def plot_signal_file(csv_path, sig):
    """为给定 CSV 生成 4×1 信号图，返回保存路径或 None。

    图与原始 CSV 同目录，命名 <stem>_signal.png。
    """
    csv_p = Path(csv_path)
    out = csv_p.with_name(csv_p.stem + '_signal.png')
    return plot_signals(sig, out_path=str(out))
