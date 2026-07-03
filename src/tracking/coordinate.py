"""坐标转换模块：像素坐标 ↔ 相机坐标系。

支持两种路径：
  1. 新路径（推荐）：已知轴径自标定 scale = D/(2R)，用比例尺直接换算
     X_m = scale * (u - cx)，不依赖物距 Z 和焦距 fx。
  2. 旧路径：Z-based 小孔相机模型 X = Z · K⁻¹ · [u,v,1]ᵀ（保留向后兼容）
"""

import numpy as np
from typing import Tuple


def pixel_to_camera(
    pixel_x: float,
    pixel_y: float,
    K_inv: np.ndarray,
    depth_z: float
) -> Tuple[float, float]:
    """将像素坐标转换为相机坐标系下的 (X, Y)。

    基于小孔相机模型：
        [X, Y, Z]ᵀ = Z · K⁻¹ · [u, v, 1]ᵀ

    其中：
        K 为相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        Z 为预设深度（相机到目标的距离）

    Args:
        pixel_x: 像素 x 坐标
        pixel_y: 像素 y 坐标
        K_inv: 相机内参矩阵的逆 (3×3)
        depth_z: 深度 Z（米）

    Returns:
        (X, Y): 相机坐标系下的物理坐标（米）
    """
    pixel = np.array([pixel_x, pixel_y, 1.0])
    cam_coord = depth_z * (K_inv @ pixel)
    return float(cam_coord[0]), float(cam_coord[1])


def pixels_to_camera_batch(
    pixel_coords: np.ndarray,
    K_inv: np.ndarray,
    depth_z: float
) -> np.ndarray:
    """批量将像素坐标转换为相机坐标系。

    Args:
        pixel_coords: (N, 2) 或 (N, 3) 数组，每行 [x, y] 或 [x, y, 1]
        K_inv: 相机内参矩阵的逆 (3×3)
        depth_z: 深度 Z（米）

    Returns:
        (N, 2) 相机坐标系下的 (X, Y) 坐标（米）
    """
    if pixel_coords.shape[1] == 2:
        ones = np.ones((pixel_coords.shape[0], 1))
        pixel_homo = np.hstack([pixel_coords, ones])
    else:
        pixel_homo = pixel_coords

    cam_coords = depth_z * (pixel_homo @ K_inv.T)
    return cam_coords[:, :2]


# =============================================================================
# 新路径：轴径自标定比例尺（不依赖 Z 和 fx）
# =============================================================================

def compute_scale_m_per_px(shaft_diameter_m: float, radius_px: float) -> float:
    """由已知轴直径和检测到的像素半径反推比例尺 (m/px)。

    推导：
        Z = (D_m · fx) / (2 · R_px)          → 代入 X = Z·(u-cx)/fx
        X_m = D_m · (u - cx) / (2 · R_px)   → scale = D_m/(2·R_px)

    Args:
        shaft_diameter_m: 电机轴直径 (米)，如 0.012
        radius_px: 检测到的轴像素半径

    Returns:
        比例尺 scale (m/px)，即每个像素对应多少米

    Raises:
        ValueError: radius_px <= 0
    """
    if radius_px <= 0:
        raise ValueError(f"像素半径无效: {radius_px}")
    return shaft_diameter_m / (2.0 * radius_px)


def pixel_to_mm(
    pixel_x: float,
    pixel_y: float,
    cx: float,
    cy: float,
    scale_m_per_px: float
) -> Tuple[float, float]:
    """用比例尺把像素坐标换算为相对主点的物理位移 (m)。

    公式：
        X_m = scale * (u - cx)
        Y_m = scale * (v - cy)

    其中 cx/cy 是相机主点（来自内参 K），scale 由 compute_scale_m_per_px() 算出。

    注意：振动峰值振幅 Amplitdue = max(X)-min(X) 与 cx/cy 无关（差分量消去），
    cx/cy 仅影响均值 mean_x/mean_y 的物理含义。

    Args:
        pixel_x: 像素 x 坐标
        pixel_y: 像素 y 坐标
        cx: 主点 cx（来自相机内参 K[0,2]）
        cy: 主点 cy（来自相机内参 K[1,2]）
        scale_m_per_px: 比例尺 (m/px)

    Returns:
        (X_m, Y_m): 相对主点的物理位移，单位 米
    """
    return (pixel_x - cx) * scale_m_per_px, (pixel_y - cy) * scale_m_per_px
