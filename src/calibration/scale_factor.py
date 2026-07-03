"""尺度因子计算模块。

从原始脚本 world s.py 中提取的尺度因子计算逻辑。
在已知相机内参和外参的情况下，逐图像计算像素坐标到世界坐标的尺度因子。
"""

import os
import numpy as np
from itertools import combinations
from typing import List, Dict


def compute_per_image_scale_factors(
    imgpoints: List[np.ndarray],
    objp: np.ndarray,
    mtx: np.ndarray,
    rvecs: List[np.ndarray],
    image_paths: List[str],
    verbose: bool = True
) -> Dict[str, float]:
    """逐图像计算世界坐标的尺度因子。

    核心公式：
        scale_x = [(fx*r11 + cx*r31)*ΔX + (fx*r12 + cx*r32)*ΔY] / Δu
        scale_y = [(fy*r21 + cy*r31)*ΔX + (fy*r22 + cy*r32)*ΔY] / Δv
        scale   = mean(scale_x, scale_y) for all corner pairs

    Args:
        imgpoints: 每张图像的 2D 角点坐标列表
        objp: 3D 世界坐标模板
        mtx: 相机内参矩阵 (3×3)
        rvecs: 旋转向量列表
        image_paths: 图像路径列表（用于提取文件名）
        verbose: 是否打印每张图像的尺度因子

    Returns:
        {filename: average_scale_factor} 字典
    """
    f_x, f_y = mtx[0, 0], mtx[1, 1]
    c_x, c_y = mtx[0, 2], mtx[1, 2]

    scale_results: Dict[str, float] = {}

    for img_idx in range(len(imgpoints)):
        pixel_points = imgpoints[img_idx]
        rvec = rvecs[img_idx]

        # 旋转向量 → 旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r11, r12, r13 = rotation_matrix[0]
        r21, r22, r23 = rotation_matrix[1]
        r31, r32, r33 = rotation_matrix[2]

        scale_factors = []

        # 遍历所有角点对
        for i, j in combinations(range(len(pixel_points)), 2):
            pixel_x_diff = pixel_points[j][0][0] - pixel_points[i][0][0]
            pixel_y_diff = pixel_points[j][0][1] - pixel_points[i][0][1]

            world_x_diff = objp[j][0] - objp[i][0]
            world_y_diff = objp[j][1] - objp[i][1]

            if pixel_x_diff != 0 and pixel_y_diff != 0:
                numerator_x = (
                    (f_x * r11 + c_x * r31) * world_x_diff +
                    (f_x * r12 + c_x * r32) * world_y_diff
                )
                numerator_y = (
                    (f_y * r21 + c_y * r31) * world_x_diff +
                    (f_y * r22 + c_y * r32) * world_y_diff
                )

                scale_factor_x = numerator_x / pixel_x_diff
                scale_factor_y = numerator_y / pixel_y_diff

                scale_factors.append((scale_factor_x + scale_factor_y) / 2)

        if scale_factors:
            avg_scale = float(np.mean(scale_factors))
            filename = os.path.basename(image_paths[img_idx])
            scale_results[filename] = avg_scale

            if verbose:
                print(f"图片 {filename} 的平均尺度因子: {avg_scale:.6f}")

    return scale_results


# 需要显式 import cv2 用于 Rodrigues
import cv2
