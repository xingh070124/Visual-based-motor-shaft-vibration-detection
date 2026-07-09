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


# =============================================================================
# Phase 8: 透视修正的各向同性 scale（推导6 — 修正透视膨胀）
# =============================================================================

def compute_perspective_corrected_scale(
    shaft_diameter_m: float,
    a_px: float,
    b_px: float
) -> float:
    """透视修正的比例尺 (m/px)。

    在透视投影下，倾斜圆投影为椭圆时，长短轴均被透视效应膨胀。
    弱透视模型假设 a = F·R/Z₀，但实际 a = F·R/(Z₀·cos²θ)，
    导致 D/(2a) 低估了真实 scale 达 cos²θ 倍。

    推导（透视投影下）：
        a = F·R / (Z₀·cos²θ)     （长轴，含透视膨胀）
        b = F·R / (Z₀·cos θ)      （短轴，含透视膨胀）
        cos θ = b / a              （倾角反演关系不变）

    由 b = F·R/(Z₀·cos θ) 得：
        Z₀/F = R·cos θ / b = (D/2)·(b/a) / b = D / (2a)·(1/cos θ)
        但 1/cos θ = a/b，所以：
        Z₀/F = D·a / (2·b²)

    验证：θ=20°, D=12mm, a=79.60px, b=74.80px
        scale = 0.012 × 79.60 / (2 × 74.80²) = 85.36 μm/px
        Z₀/F = 0.25/2928.5 = 85.36 μm/px ✓

    关键结论：位移换算系数在所有方向相同（均为 Z₀/F），
    倾斜只改变投影形状（圆→椭圆），不改变位移的 scale。
    各向异性换算（不同方向不同 scale）在位移测量中是错误的。

    Args:
        shaft_diameter_m: 电机轴直径 (米)
        a_px: 椭圆半长轴 (像素)
        b_px: 椭圆半短轴 (像素)

    Returns:
        透视修正的各向同性比例尺 (m/px)

    Raises:
        ValueError: a_px 或 b_px <= 0
    """
    if a_px <= 0 or b_px <= 0:
        raise ValueError(f"椭圆半轴无效: a={a_px}, b={b_px}")
    return shaft_diameter_m * a_px / (2.0 * b_px * b_px)


# =============================================================================
# Phase 5: 各向异性坐标换算 — 已废弃（保留向后兼容）
# =============================================================================

def estimate_tilt_angle(a: float, b: float) -> float:
    """从椭圆长短轴反演倾角（推导2）。

    公式：theta = arccos(b/a)

    Args:
        a: 椭圆半长轴（像素）
        b: 椭圆半短轴（像素）

    Returns:
        倾角（度），范围 [0, 90)
    """
    if a <= 0 or b <= 0:
        return 0.0
    ratio = min(a, b) / max(a, b)
    ratio = min(ratio, 1.0)  # 防止浮点误差导致 >1
    return float(np.degrees(np.arccos(ratio)))


def pixel_to_mm_anisotropic(
    pixel_x: float,
    pixel_y: float,
    cx: float,
    cy: float,
    scale_major: float,
    scale_minor: float,
    ellipse_angle: float
) -> Tuple[float, float]:
    """各向异性 scale 坐标换算（推导3）。

    倾斜轴投影为椭圆后，长短轴方向缩放不同：
      - 长轴方向（⊥倾斜）：scale_major = D/(2a)
      - 短轴方向（∥倾斜）：scale_minor = D/(2b)

    流程：
      1. 像素位移 (du, dv) = (pixel_x - cx, pixel_y - cy)
      2. 旋转到椭圆主轴坐标系
      3. 分别用 scale_major / scale_minor 换算
      4. 旋转回相机坐标系

    Args:
        pixel_x: 像素 x 坐标
        pixel_y: 像素 y 坐标
        cx: 主点 cx
        cy: 主点 cy
        scale_major: 长轴方向比例尺 (m/px) = D/(2a)
        scale_minor: 短轴方向比例尺 (m/px) = D/(2b)
        ellipse_angle: 椭圆长轴方向角（度，来自 cv2.fitEllipse）

    Returns:
        (X_m, Y_m): 相对主点的物理位移，单位 米
    """
    rad = np.radians(ellipse_angle)
    du = pixel_x - cx
    dv = pixel_y - cy

    # 旋转到椭圆主轴坐标系
    du_major = du * np.cos(rad) + dv * np.sin(rad)
    dv_minor = -du * np.sin(rad) + dv * np.cos(rad)

    # 各向异性换算
    X_major = scale_major * du_major
    Y_minor = scale_minor * dv_minor

    # 旋转回相机坐标系
    X = X_major * np.cos(rad) - Y_minor * np.sin(rad)
    Y = X_major * np.sin(rad) + Y_minor * np.cos(rad)

    return float(X), float(Y)
