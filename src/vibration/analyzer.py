"""振动幅度分析模块。

从相机坐标系下的运动轨迹数据计算电机轴的振动幅度。
"""

import numpy as np
from typing import Dict
from scipy import signal as sig


def remove_drift(signal: np.ndarray, method: str = 'linear') -> np.ndarray:
    """去除低频漂移（论文H: AR(1)漂移模型）。

    Args:
        signal: 一维信号序列
        method: 'linear' 线性去趋势（AR(1)一阶近似）
                'poly2' 二阶多项式去趋势
                'none' 不去趋势

    Returns:
        去趋势后的信号
    """
    if method == 'none' or len(signal) < 3:
        return signal
    elif method == 'linear':
        return sig.detrend(signal, type='linear')
    elif method == 'poly2':
        t = np.arange(len(signal))
        coeffs = np.polyfit(t, signal, 2)
        trend = np.polyval(coeffs, t)
        return signal - trend
    return signal


def compute_vibration_amplitude(
    cam_x: np.ndarray,
    cam_y: np.ndarray,
    detrend: bool = True
) -> Dict[str, float]:
    """根据相机坐标系下的 (X, Y) 轨迹计算振动幅度。

    振动定义为峰峰值（peak-to-peak）：
        amplitude = max(value) - min(value)

    总幅度为 X 和 Y 方向振幅的欧氏距离：
        amplitude_total = sqrt(amplitude_x² + amplitude_y²)

    Args:
        cam_x: X 方向位移序列（米）
        cam_y: Y 方向位移序列（米）
        detrend: 是否去除低频漂移（论文H: AR(1)模型，默认开启）

    Returns:
        {
            'amplitude_x': X 方向峰峰振幅 (m),
            'amplitude_y': Y 方向峰峰振幅 (m),
            'amplitude_total': 总振幅 (m),
            'amplitude_x_mm': X 方向峰峰振幅 (mm),
            'amplitude_y_mm': Y 方向峰峰振幅 (mm),
            'amplitude_total_mm': 总振幅 (mm),
            'amplitude_x_detrended_mm': 去漂移后 X 振幅 (mm),
            'amplitude_y_detrended_mm': 去漂移后 Y 振幅 (mm),
            'amplitude_total_detrended_mm': 去漂移后总振幅 (mm),
            'mean_x': X 方向均值 (m),
            'mean_y': Y 方向均值 (m),
            'std_x': X 方向标准差 (m),
            'std_y': Y 方向标准差 (m),
            'drift_x_mm': X 方向漂移量 (mm),
            'drift_y_mm': Y 方向漂移量 (mm),
            'n_samples': 样本数,
        }
    """
    if len(cam_x) == 0 or len(cam_y) == 0:
        raise ValueError("输入数据为空")

    # 原始振幅（含漂移）
    amplitude_x = float(np.max(cam_x) - np.min(cam_x))
    amplitude_y = float(np.max(cam_y) - np.min(cam_y))
    amplitude_total = float(np.sqrt(amplitude_x ** 2 + amplitude_y ** 2))

    # 去漂移后振幅（论文H: AR(1)漂移去除）
    if detrend:
        cam_x_dt = remove_drift(cam_x, 'linear')
        cam_y_dt = remove_drift(cam_y, 'linear')
        amp_x_dt = float(np.max(cam_x_dt) - np.min(cam_x_dt))
        amp_y_dt = float(np.max(cam_y_dt) - np.min(cam_y_dt))
        amp_total_dt = float(np.sqrt(amp_x_dt ** 2 + amp_y_dt ** 2))
    else:
        amp_x_dt = amplitude_x
        amp_y_dt = amplitude_y
        amp_total_dt = amplitude_total

    # 漂移量（后1/3均值 - 前1/3均值）
    n = len(cam_x)
    third = max(1, n // 3)
    drift_x = float(np.mean(cam_x[int(2 * n / 3):]) - np.mean(cam_x[:third]))
    drift_y = float(np.mean(cam_y[int(2 * n / 3):]) - np.mean(cam_y[:third]))

    return {
        'amplitude_x': amplitude_x,
        'amplitude_y': amplitude_y,
        'amplitude_total': amplitude_total,
        'amplitude_x_mm': amplitude_x * 1000,
        'amplitude_y_mm': amplitude_y * 1000,
        'amplitude_total_mm': amplitude_total * 1000,
        'amplitude_x_detrended_mm': amp_x_dt * 1000,
        'amplitude_y_detrended_mm': amp_y_dt * 1000,
        'amplitude_total_detrended_mm': amp_total_dt * 1000,
        'mean_x': float(np.mean(cam_x)),
        'mean_y': float(np.mean(cam_y)),
        'std_x': float(np.std(cam_x)),
        'std_y': float(np.std(cam_y)),
        'drift_x_mm': drift_x * 1000,
        'drift_y_mm': drift_y * 1000,
        'n_samples': len(cam_x),
    }


def print_vibration_result(result: Dict[str, float]):
    """格式化打印振动分析结果。"""
    print("\n" + "=" * 50)
    print("振动分析结果")
    print("=" * 50)
    print(f"  样本数:           {result['n_samples']}")
    print(f"  X 方向振幅:       {result['amplitude_x_mm']:.4f} mm")
    print(f"  Y 方向振幅:       {result['amplitude_y_mm']:.4f} mm")
    print(f"  总振动幅度:       {result['amplitude_total_mm']:.4f} mm")
    print(f"  去漂移后 X 振幅:  {result.get('amplitude_x_detrended_mm', 0):.4f} mm")
    print(f"  去漂移后 Y 振幅:  {result.get('amplitude_y_detrended_mm', 0):.4f} mm")
    print(f"  去漂移后总振幅:   {result.get('amplitude_total_detrended_mm', 0):.4f} mm")
    print(f"  X 漂移量:         {result.get('drift_x_mm', 0):.4f} mm")
    print(f"  Y 漂移量:         {result.get('drift_y_mm', 0):.4f} mm")
    print(f"  X 均值:           {result['mean_x'] * 1000:.4f} mm")
    print(f"  Y 均值:           {result['mean_y'] * 1000:.4f} mm")
    print(f"  X 标准差:         {result['std_x'] * 1000:.4f} mm")
    print(f"  Y 标准差:         {result['std_y'] * 1000:.4f} mm")
    print("=" * 50)
