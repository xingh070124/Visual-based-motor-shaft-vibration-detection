"""集中配置：所有硬编码参数统一管理。

修改本文件即可调整相机内参、跟踪参数、标定参数等，无需改动业务代码。
"""

import os
import numpy as np
import cv2

# =============================================================================
# 路径配置
# =============================================================================

# STARK 代码库根目录（外部依赖）
STARK_ROOT = r"D:\github\clone\Stark"

# 数据目录（相对于项目根目录）
IMAGE_FOLDER_TRACKING = "frames200-1"        # 跟踪用帧序列目录
IMAGE_FOLDER_CALIB_1 = "./images001"          # 标定图像目录1
IMAGE_FOLDER_CALIB_2 = "./images002"          # 标定图像目录2
IMAGE_FOLDER_UNDISTORT = "./images2"          # 去畸变测试图像目录

# STARK 预训练权重路径（相对于 STARK_ROOT 的 save_dir）
CHECKPOINT_REL_PATH = "checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar"
CHECKPOINT_PARAM_NAME = "baseline"
CHECKPOINT_DATASET_NAME = "OTB100"

# 输出文件
OUTPUT_EXCEL = "stark_tracking_camera_coords.xlsx"

# =============================================================================
# 相机内参 — 一倍（默认）
# =============================================================================

FX_1X = 871.634556391140
FY_1X = 1159.30440189264
CX_1X = 375.114519279214
CY_1X = 210.871016530132

K_1X = np.array([
    [FX_1X, 0, CX_1X],
    [0, FY_1X, CY_1X],
    [0, 0, 1]
])
K_INV_1X = np.linalg.inv(K_1X)

# 相机内参 — test/figures 标定结果（2026-07-02，22张图，9×6棋盘格，重投影误差 0.076px）
FX_CALIB_NEW = 2928.5148
FY_CALIB_NEW = 2921.1419
CX_CALIB_NEW = 1013.4001
CY_CALIB_NEW = 757.2565

K_CALIB_NEW = np.array([
    [FX_CALIB_NEW, 0, CX_CALIB_NEW],
    [0, FY_CALIB_NEW, CY_CALIB_NEW],
    [0, 0, 1]
])
K_INV_CALIB_NEW = np.linalg.inv(K_CALIB_NEW)

# 相机内参 — 二倍（备用，注释掉）
# FX_2X = 1791.22202940993
# FY_2X = 2397.85179602954
# CX_2X = 230.701118313539
# CY_2X = 185.173943174303
# K_2X = np.array([[FX_2X, 2.18133573006278, CX_2X], [0, FY_2X, CY_2X], [0, 0, 1]])
# K_INV_2X = np.linalg.inv(K_2X)

# 当前使用的内参（默认使用新标定结果）
K = K_CALIB_NEW
K_INV = K_INV_CALIB_NEW

# 主点坐标（像素坐标系原点在图像中的位置），供比例尺换算 X_mm = scale*(u-cx) 使用
CX = float(K[0, 2])
CY = float(K[1, 2])

# =============================================================================
# 深度与物理参数
# =============================================================================

# 目标深度（米）
# DEPRECATED: 坐标转换已改用轴径自标定 scale，不再依赖此值。
# 仅历史保留，如需访问旧 Z-based 函数仍可引用。
DEPTH_Z = 0.1817

# 电机轴直径（米）
SHAFT_DIAMETER_M = 0.012  # 12mm
SHAFT_DIAMETER_MM = SHAFT_DIAMETER_M * 1000  # 12.0 mm（打印/调试用）

# 比例尺标定：取前 N 帧有效检测半径的中位数计算 scale = D/(2*R)
SCALE_CALIB_FRAMES = 50

# =============================================================================
# 轮廓检测参数
# =============================================================================

# 圆度阈值（越接近1越圆）
CIRCULARITY_THRESHOLD = 0.5

# 半径偏差允许范围（相对于预期半径的比例）
RADIUS_TOLERANCE = 0.3  # ±30%

# 最小轮廓面积（像素）
MIN_CONTOUR_AREA = 100

# Canny 边缘检测阈值
CANNY_LOW = 50
CANNY_HIGH = 150

# 高斯模糊核大小
GAUSSIAN_BLUR_KERNEL = (5, 5)

# CLAHE 光照自适应预处理参数（增强光照鲁棒性）
CLAHE_CLIPLIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Hough 圆检测参数（改进后）
HOUGH_PARAM1 = 100       # 内部 Canny 高阈值
HOUGH_PARAM2 = 40        # 累加器阈值（原25→40，减少假阳性）
HOUGH_MIN_RADIUS_RATIO = 0.85   # 最小半径比例（原0.4→0.85）
HOUGH_MAX_RADIUS_RATIO = 1.15   # 最大半径比例（原1.8→1.15）

# 滑动窗口中位数参数（替代逐帧自适应更新）
RADIUS_HISTORY_WINDOW = 30       # 滑动窗口大小（帧数）
RADIUS_HISTORY_MIN = 10          # 最少需要多少帧才开始用中位数

# 半径异常检测阈值（偏离首帧检测半径的比例）
RADIUS_ANOMALY_THRESHOLD = 0.30  # 偏离30%视为异常

# =============================================================================
# 相机标定参数
# =============================================================================

# 棋盘格内角点数 — 大棋盘格 (cameraCalibration.py)
CHECKERBOARD_LARGE = (9, 6)

# 棋盘格内角点数 — 小棋盘格 (world s.py)
CHECKERBOARD_SMALL = (5, 3)

# 小棋盘格方格边长（米）
SQUARE_SIZE_SMALL = 0.006  # 6mm

# 角点优化终止条件
CRITERIA_CALIB = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# =============================================================================
# 可视化参数
# =============================================================================

# 显示窗口名称
WINDOW_SELECT_TARGET = "Select Target"
WINDOW_TRACKING = "Tracking"

# 跟踪可视化颜色（BGR）
COLOR_BBOX = (0, 255, 0)      # 绿色 bounding box
COLOR_CENTER = (0, 0, 255)    # 红色中心点
COLOR_TEXT = (255, 255, 255)  # 白色文字

# 帧显示等待时间（ms），0=无限等待
WAIT_KEY_DELAY = 30

# ESC 键退出
QUIT_KEY = 27

# =============================================================================
# 激光数据处理参数
# =============================================================================

# 脚本 yuanqiujie.py 的参数
LASER_OFFSET_B = 0.0
LASER_OFFSET_C = 1.02436125638347

# 脚本 V (xy m---mm and pingyi).py 的参数
LASER_SCALE_B = 1.0
LASER_SCALE_C = 1.3


# =============================================================================
# 辅助函数
# =============================================================================

def get_expected_radius_pixels(fx: float = None, z: float = None, diameter_m: float = None) -> float:
    """根据相机内参和深度计算预期的电机轴像素半径。

    DEPRECATED: 比例尺已改为轴径自标定（ROI 先验 + 前 N 帧中位数），
    不再需要此函数。推荐使用 shaft_detector.estimate_expected_radius_from_roi()。

    Args:
        fx: 相机焦距 fx，默认使用配置中的 FX_1X
        z: 深度（米），默认使用配置中的 DEPTH_Z
        diameter_m: 轴直径（米），默认使用配置中的 SHAFT_DIAMETER_M

    Returns:
        预期像素半径
    """
    if fx is None:
        fx = FX_1X
    if z is None:
        z = DEPTH_Z
    if diameter_m is None:
        diameter_m = SHAFT_DIAMETER_M
    return (diameter_m / 2 * fx) / z


# =============================================================================
# 批量处理配置
# =============================================================================

# 测试视频目录
VIDEO_FOLDER = "test/video"

# 批量处理输出目录
BATCH_OUTPUT_DIR = "test/output"

# 输出视频 FPS（None 表示保持原始 FPS）
OUTPUT_VIDEO_FPS = None


def get_project_root() -> str:
    """返回项目根目录（CLAUDE.md 所在目录）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
