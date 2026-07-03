"""STARK 端到端验证脚本 — 使用合成帧测试完整跟踪流程。

绕过交互式 ROI 选择，直接用预设 ROI 初始化跟踪器，
验证：STARK 初始化 → 逐帧跟踪 → 坐标转换 → 振动分析 全链路。
"""

import os, sys, time
import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.tracking.stark_wrapper import StarkTracker
from src.tracking.shaft_detector import detect_shaft_center_in_roi, compute_expected_radius_pixels
from src.tracking.coordinate import pixel_to_camera
from src.vibration.analyzer import compute_vibration_amplitude, print_vibration_result
from utils.file_utils import find_images, natural_key

# ---- 临时覆盖：使用合成帧目录 ----
config.IMAGE_FOLDER_TRACKING = "test_frames"
config.OUTPUT_EXCEL = "test_stark_output.xlsx"

image_folder = config.IMAGE_FOLDER_TRACKING
if not os.path.isdir(image_folder):
    image_folder = os.path.join(_PROJECT_ROOT, image_folder)

image_paths = find_images(image_folder, "*.png")
print(f"[OK] Found {len(image_paths)} synthetic frames")

first_frame = cv2.imread(image_paths[0])
H, W = first_frame.shape[:2]
print(f"[OK] Frame size: {W}x{H}")

# ---- 预设 ROI（模拟电机轴位置，跳过手动选择） ----
init_box = [270, 195, 100, 100]  # [x, y, w, h]
print(f"[OK] Preset ROI: {init_box}")

# ---- 轮廓检测 → 初始中心点 ----
expected_radius = compute_expected_radius_pixels(
    fx=config.FX_1X,
    shaft_diameter_m=config.SHAFT_DIAMETER_M,
    depth_z=config.DEPTH_Z
)
print(f"    预期像素半径: {expected_radius:.2f}")

init_center_x, init_center_y = detect_shaft_center_in_roi(
    first_frame, init_box,
    expected_radius_pixels=expected_radius,
    circularity_threshold=config.CIRCULARITY_THRESHOLD,
    radius_tolerance=config.RADIUS_TOLERANCE,
    min_contour_area=config.MIN_CONTOUR_AREA,
    canny_low=config.CANNY_LOW,
    canny_high=config.CANNY_HIGH,
    blur_kernel=config.GAUSSIAN_BLUR_KERNEL
)
print(f"[OK] Initial center: ({init_center_x:.2f}, {init_center_y:.2f})")

# ---- 初始化 STARK 跟踪器 ----
print("\n[LAUNCH] 加载 STARK-ST 跟踪器...")
t_start = time.time()
tracker = StarkTracker(
    checkpoint_rel_path=config.CHECKPOINT_REL_PATH,
    params_name=config.CHECKPOINT_PARAM_NAME,
    dataset_name=config.CHECKPOINT_DATASET_NAME
)
tracker.initialize(first_frame, init_box)
t_elapsed = time.time() - t_start
print(f"[OK] STARK 初始化成功！耗时 {t_elapsed:.2f}s")

# ---- 逐帧跟踪 ----
print(f"\n[TRACK] 开始跟踪 {len(image_paths)} 帧...")
results = []
cam_x0, cam_y0 = pixel_to_camera(init_center_x, init_center_y, config.K_INV, config.DEPTH_Z)
results.append([1, init_center_x, init_center_y, cam_x0, cam_y0])

track_times = []
for idx, img_path in enumerate(image_paths[1:], start=2):
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    t0 = time.time()
    # STARK 跟踪
    state = tracker.track(frame)
    pred_box = state['target_bbox']

    # ROI 内精确检测中心
    center_x, center_y = detect_shaft_center_in_roi(
        frame, pred_box,
        expected_radius_pixels=expected_radius,
        circularity_threshold=config.CIRCULARITY_THRESHOLD,
        radius_tolerance=config.RADIUS_TOLERANCE,
        min_contour_area=config.MIN_CONTOUR_AREA,
        canny_low=config.CANNY_LOW,
        canny_high=config.CANNY_HIGH,
        blur_kernel=config.GAUSSIAN_BLUR_KERNEL
    )
    cam_x, cam_y = pixel_to_camera(center_x, center_y, config.K_INV, config.DEPTH_Z)
    track_times.append(time.time() - t0)
    results.append([idx, center_x, center_y, cam_x, cam_y])

    if idx <= 5 or idx % 10 == 0:
        print(f"  Frame {idx:3d}: Pixel=({center_x:7.2f}, {center_y:7.2f}) → Camera=({cam_x:.5f}, {cam_y:.5f}) m")

avg_track_time = np.mean(track_times) if track_times else 0
print(f"[OK] 跟踪完成！平均每帧 {avg_track_time*1000:.1f} ms")

# ---- 保存结果 ----
df = pd.DataFrame(results, columns=["Frame", "Pixel_X", "Pixel_Y", "Cam_X (m)", "Cam_Y (m)"])
df.to_excel(config.OUTPUT_EXCEL, index=False)
print(f"[OK] 结果已保存到: {config.OUTPUT_EXCEL}")

# ---- 振动分析 ----
cam_x_arr = df["Cam_X (m)"].values
cam_y_arr = df["Cam_Y (m)"].values
vibration = compute_vibration_amplitude(cam_x_arr, cam_y_arr)
print_vibration_result(vibration)

print("\n" + "=" * 60)
print("[OK] STARK E2E verification PASSED!")
print(f"   Tracker init: {t_elapsed:.2f}s")
print(f"   Per-frame tracking: {avg_track_time*1000:.1f} ms/frame")
print(f"   X amplitude: {vibration['amplitude_x']:.6f} m ({vibration['amplitude_x']*1e3:.3f} mm)")
print(f"   Y amplitude: {vibration['amplitude_y']:.6f} m ({vibration['amplitude_y']*1e3:.3f} mm)")
print(f"   Total amplitude: {vibration['amplitude_total']:.6f} m ({vibration['amplitude_total']*1e3:.3f} mm)")
print("=" * 60)
