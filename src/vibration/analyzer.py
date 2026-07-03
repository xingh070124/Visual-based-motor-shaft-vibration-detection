"""振动幅度分析模块。

从相机坐标系下的运动轨迹数据计算电机轴的振动幅度。
"""

import numpy as np
from typing import Dict


def compute_vibration_amplitude(
    cam_x: np.ndarray,
    cam_y: np.ndarray
) -> Dict[str, float]:
    """根据相机坐标系下的 (X, Y) 轨迹计算振动幅度。

    振动定义为峰峰值（peak-to-peak）：
        amplitude = max(value) - min(value)

    总幅度为 X 和 Y 方向振幅的欧氏距离：
        amplitude_total = sqrt(amplitude_x² + amplitude_y²)

    Args:
        cam_x: X 方向位移序列（米）
        cam_y: Y 方向位移序列（米）

    Returns:
        {
            'amplitude_x': X 方向峰峰振幅 (m),
            'amplitude_y': Y 方向峰峰振幅 (m),
            'amplitude_total': 总振幅 (m),
            'amplitude_x_mm': X 方向峰峰振幅 (mm),
            'amplitude_y_mm': Y 方向峰峰振幅 (mm),
            'amplitude_total_mm': 总振幅 (mm),
            'mean_x': X 方向均值 (m),
            'mean_y': Y 方向均值 (m),
            'std_x': X 方向标准差 (m),
            'std_y': Y 方向标准差 (m),
            'n_samples': 样本数,
        }
    """
    if len(cam_x) == 0 or len(cam_y) == 0:
        raise ValueError("输入数据为空")

    amplitude_x = float(np.max(cam_x) - np.min(cam_x))
    amplitude_y = float(np.max(cam_y) - np.min(cam_y))
    amplitude_total = float(np.sqrt(amplitude_x ** 2 + amplitude_y ** 2))

    return {
        'amplitude_x': amplitude_x,
        'amplitude_y': amplitude_y,
        'amplitude_total': amplitude_total,
        'amplitude_x_mm': amplitude_x * 1000,
        'amplitude_y_mm': amplitude_y * 1000,
        'amplitude_total_mm': amplitude_total * 1000,
        'mean_x': float(np.mean(cam_x)),
        'mean_y': float(np.mean(cam_y)),
        'std_x': float(np.std(cam_x)),
        'std_y': float(np.std(cam_y)),
        'n_samples': len(cam_x),
    }


def print_vibration_result(result: Dict[str, float]):
    """格式化打印振动分析结果。"""
    print("\n" + "=" * 50)
    print("振动分析结果")
    print("=" * 50)
    print(f"  样本数:         {result['n_samples']}")
    print(f"  X 方向振幅:     {result['amplitude_x_mm']:.4f} mm")
    print(f"  Y 方向振幅:     {result['amplitude_y_mm']:.4f} mm")
    print(f"  总振动幅度:     {result['amplitude_total_mm']:.4f} mm")
    print(f"  X 均值:         {result['mean_x'] * 1000:.4f} mm")
    print(f"  Y 均值:         {result['mean_y'] * 1000:.4f} mm")
    print(f"  X 标准差:       {result['std_x'] * 1000:.4f} mm")
    print(f"  Y 标准差:       {result['std_y'] * 1000:.4f} mm")
    print("=" * 50)
