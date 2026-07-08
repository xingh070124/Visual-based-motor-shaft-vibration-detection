"""电机轴圆形轮廓检测模块。

在跟踪器预测的 ROI 区域内，通过 Canny 边缘检测 + 轮廓圆度筛选，
精确定位电机轴的中心坐标。

提取自原始脚本 stark_tracking_camera_coords.py 的 detect_shaft_center_in_roi() 函数。
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def estimate_expected_radius_from_roi(roi_box) -> float:
    """从用户框选的 ROI 尺寸估算电机轴像素半径。

    用于替代 Z-based compute_expected_radius_pixels() 作为检测先验。
    假设 ROI 框大致贴合电机轴边缘 → 轴半径 ≈ min(W, H) / 2。

    Args:
        roi_box: (x, y, w, h) — 可以是 tuple/list/ndarray

    Returns:
        估算的像素半径（float）。返回值 > 0。
    """
    x, y, w, h = roi_box
    return min(abs(w), abs(h)) / 2.0


def detect_shaft_center_in_roi(
    frame: np.ndarray,
    roi_box: Tuple[float, float, float, float],
    expected_radius_pixels: float,
    circularity_threshold: float = 0.5,
    radius_tolerance: float = 0.3,
    min_contour_area: float = 100.0,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: Tuple[int, int] = (5, 5),
    return_radius: bool = False
):
    """在给定的 ROI 中检测电机轴圆形轮廓，返回圆心坐标。

    处理流程：
        1. 裁剪 ROI 区域
        2. 灰度化 + 高斯模糊
        3. Canny 边缘检测
        4. 查找轮廓，按圆度筛选
        5. 用最小包围圆获取中心，半径校验
        6. 若未找到合适轮廓，回退到 ROI 中心

    Args:
        frame: 完整帧图像 (BGR)
        roi_box: 边界框 (x, y, w, h)
        expected_radius_pixels: 预期像素半径（基于相机内参和已知轴径计算）
        circularity_threshold: 圆度阈值（0~1），越接近 1 越严格，默认 0.5
        radius_tolerance: 半径允许偏差比例，默认 0.3（±30%）
        min_contour_area: 最小轮廓面积，过滤噪点，默认 100
        canny_low: Canny 低阈值，默认 50
        canny_high: Canny 高阈值，默认 150
        blur_kernel: 高斯模糊核大小，默认 (5, 5)
        return_radius: 是否同时返回检测半径，默认 False

    Returns:
        return_radius=False: (center_x, center_y)
        return_radius=True:  (center_x, center_y, radius)
        回退情况下 radius = expected_radius_pixels
    """
    x, y, w, h = map(int, roi_box)

    # 边界检查
    h_img, w_img = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    if w <= 0 or h <= 0:
        print("[WARN]️ ROI 区域无效，回退到 bbox 中心")
        if return_radius:
            return x + w / 2, y + h / 2, 0.0
        return x + w / 2, y + h / 2

    roi = frame[y:y + h, x:x + w]

    # 预处理：灰度化 + 统一高斯模糊 + CLAHE光照自适应
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ---- 策略1: 霍夫圆检测（对视频/实拍图像更鲁棒） ----
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1,
        minDist=max(20, int(expected_radius_pixels * 1.5)),
        param1=100, param2=40,
        minRadius=int(expected_radius_pixels * 0.85),
        maxRadius=int(expected_radius_pixels * 1.15)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = None
        best_score = float('inf')
        roi_center = np.array([w / 2, h / 2])

        for c in circles[0, :]:
            cr, cc, cradius = int(c[1]), int(c[0]), int(c[2])
            # 评分: 半径偏差 + 距离 ROI 中心偏差
            r_err = abs(cradius - expected_radius_pixels) / expected_radius_pixels
            d_err = np.linalg.norm(np.array([cc, cr]) - roi_center) / max(w, h)
            score = r_err + d_err * 2
            if score < best_score:
                best_score = score
                best_circle = (cc, cr, cradius)

        if best_circle is not None and best_score < 0.8:
            cx_roi, cy_roi, radius = best_circle
            # 圆度验证：在检测到的圆周围裁剪，做轮廓圆度检查
            r_check = int(radius)
            y1 = max(0, cy_roi - r_check)
            y2 = min(h, cy_roi + r_check)
            x1 = max(0, cx_roi - r_check)
            x2 = min(w, cx_roi + r_check)
            if y2 > y1 and x2 > x1:
                circle_region = gray[y1:y2, x1:x2]
                circle_edges = cv2.Canny(circle_region, canny_low, canny_high)
                cntrs, _ = cv2.findContours(circle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cntrs:
                    best_cnt = max(cntrs, key=cv2.contourArea)
                    area = cv2.contourArea(best_cnt)
                    perim = cv2.arcLength(best_cnt, True)
                    if perim > 0:
                        circ = 4 * np.pi * area / (perim ** 2)
                        if circ > 0.3:  # 最低圆度要求
                            center_x = x + cx_roi
                            center_y = y + cy_roi
                            if return_radius:
                                return center_x, center_y, float(radius)
                            return center_x, center_y
            # 圆度验证失败，仍接受Hough结果（降级策略）
            center_x = x + cx_roi
            center_y = y + cy_roi
            if return_radius:
                return center_x, center_y, float(radius)
            return center_x, center_y

    # ---- 策略2: 轮廓圆度检测（对清晰棋盘格/理想图像） ----
    # gray 已在策略1前做过高斯模糊+CLAHE，直接用
    edges = cv2.Canny(gray, canny_low, canny_high)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        best_contour = None
        best_circularity = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > best_circularity:
                best_circularity = circularity
                best_contour = cnt

        if best_contour is not None and best_circularity > circularity_threshold:
            (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(best_contour)
            radius_diff = abs(radius - expected_radius_pixels)
            max_allowed_diff = expected_radius_pixels * radius_tolerance

            if radius_diff < max_allowed_diff:
                center_x = x + cx_roi
                center_y = y + cy_roi
                if return_radius:
                    return center_x, center_y, radius
                return center_x, center_y

    # ---- 策略3: 回退到 ROI 中心（未检测到圆） ----
    if return_radius:
        return x + w / 2, y + h / 2, 0.0
    return x + w / 2, y + h / 2


def compute_expected_radius_pixels(
    fx: float,
    shaft_diameter_m: float,
    depth_z: float
) -> float:
    """根据相机内参和物理参数计算预期的像素半径。

    DEPRECATED: 比例尺已改为轴径自标定（ROI 先验 + 前 N 帧中位数），
    不再需要此函数。推荐使用 estimate_expected_radius_from_roi() 作为初始检测先验。

    公式：radius_pixels = (shaft_diameter_m / 2 * fx) / depth_z

    Args:
        fx: 相机焦距 (像素)
        shaft_diameter_m: 电机轴直径 (米)
        depth_z: 目标深度 (米)

    Returns:
        预期像素半径
    """
    return (shaft_diameter_m / 2 * fx) / depth_z


# =============================================================================
# Phase 3: 椭圆检测 — 用于倾斜轴场景
# =============================================================================

def detect_ellipse_in_roi(
    frame: np.ndarray,
    roi_box: Tuple[float, float, float, float],
    expected_radius_pixels: float,
    circularity_threshold: float = 0.3,
    min_contour_area: float = 100.0,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: Tuple[int, int] = (5, 5),
    return_params: bool = False
):
    """在 ROI 内拟合椭圆，返回中心坐标和椭圆参数。

    用于轴不垂直于相机平面的场景：倾斜圆投影为椭圆，
    通过 fitEllipse 获取长短轴，再用各向异性 scale 补偿。

    Args:
        frame: 完整帧图像 (BGR)
        roi_box: 边界框 (x, y, w, h)
        expected_radius_pixels: 预期像素半径（基于 ROI 尺寸估算）
        circularity_threshold: 最低圆度要求，默认 0.3
        min_contour_area: 最小轮廓面积，过滤噪点
        canny_low: Canny 低阈值
        canny_high: Canny 高阈值
        blur_kernel: 高斯模糊核大小
        return_params: 是否返回完整椭圆参数 (a, b, angle)

    Returns:
        return_params=False: (center_x, center_y)
        return_params=True:  (center_x, center_y, a, b, angle)
            其中 a, b 为半轴长度（像素），angle 为长轴方向角（度）
        回退情况下 a=b=expected_radius, angle=0
    """
    x, y, w, h = map(int, roi_box)

    # 边界检查
    h_img, w_img = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    if w <= 0 or h <= 0:
        print("[WARN] ROI 区域无效，回退到 bbox 中心")
        if return_params:
            return x + w / 2, y + h / 2, expected_radius_pixels, expected_radius_pixels, 0.0
        return x + w / 2, y + h / 2

    roi = frame[y:y + h, x:x + w]

    # 预处理：灰度化 + 高斯模糊 + CLAHE
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Canny 边缘检测
    edges = cv2.Canny(gray, canny_low, canny_high)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        if return_params:
            return x + w / 2, y + h / 2, expected_radius_pixels, expected_radius_pixels, 0.0
        return x + w / 2, y + h / 2

    # 选面积最大且圆度达标的轮廓
    best_contour = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < circularity_threshold:
            continue
        # 评分：面积 × 圆度
        score = area * circularity
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None or len(best_contour) < 5:
        if return_params:
            return x + w / 2, y + h / 2, expected_radius_pixels, expected_radius_pixels, 0.0
        return x + w / 2, y + h / 2

    # fitEllipse 返回 ((cx, cy), (width, height), angle)
    # width/height 是全长轴（直径），需除以2得到半轴
    (cx_roi, cy_roi), (width, height), angle = cv2.fitEllipse(best_contour)
    a = max(width, height) / 2.0  # 半长轴
    b = min(width, height) / 2.0  # 半短轴

    # 半径校验：半长轴不应偏离预期太多
    if a > expected_radius_pixels * 1.5 or a < expected_radius_pixels * 0.5:
        print(f"[WARN] 椭圆半长轴 {a:.1f} 偏离预期 {expected_radius_pixels:.1f}，回退")
        if return_params:
            return x + w / 2, y + h / 2, expected_radius_pixels, expected_radius_pixels, 0.0
        return x + w / 2, y + h / 2

    center_x = x + cx_roi
    center_y = y + cy_roi

    if return_params:
        return center_x, center_y, a, b, angle
    return center_x, center_y
