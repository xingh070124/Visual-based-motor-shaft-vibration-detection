"""主流程入口：基于视觉的电机轴振动检测。

完整流水线：
    1. 加载图像序列
    2. 手动选择目标 ROI（首帧）
    3. 轮廓精确检测 → 初始中心点
    4. STARK-ST 跟踪器初始化
    5. 逐帧：跟踪 → 轮廓检测 → 坐标转换
    6. 振动幅度计算
    7. 结果保存到 Excel

用法：
    python -m src.main
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.tracking.stark_wrapper import StarkTracker
from src.tracking.shaft_detector import (
    detect_shaft_center_in_roi,
    estimate_expected_radius_from_roi
)
from src.tracking.coordinate import compute_scale_m_per_px, pixel_to_mm
from src.vibration.analyzer import (
    compute_vibration_amplitude,
    print_vibration_result
)
from utils.file_utils import find_images, natural_key


def main():
    # =====================================================================
    # 1. 加载图像序列
    # =====================================================================
    image_folder = config.IMAGE_FOLDER_TRACKING
    if not os.path.isdir(image_folder):
        # 尝试相对于项目根目录
        image_folder = os.path.join(
            os.path.dirname(_PROJECT_ROOT), image_folder
        )
    image_paths = find_images(image_folder, "*.png")
    if not image_paths:
        print(f"[ERROR] No PNG images found in: {image_folder}")
        return

    print(f"[OK] Found {len(image_paths)} images")

    # =====================================================================
    # 2. 加载首帧 & 手动选择 ROI
    # =====================================================================
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print(f"[ERROR] Cannot read first frame: {image_paths[0]}")
        return

    cv2.namedWindow(config.WINDOW_SELECT_TARGET, cv2.WINDOW_KEEPRATIO)
    init_box = cv2.selectROI(
        config.WINDOW_SELECT_TARGET, first_frame, False, False
    )
    cv2.destroyWindow(config.WINDOW_SELECT_TARGET)

    if sum(init_box) == 0:
        print("[ERROR] 目标选择已取消")
        return

    init_box = [int(v) for v in init_box]
    print(f"[OK] 初始 ROI: {init_box}")

    # =====================================================================
    # 3. 轮廓精确检测 → 初始中心点 & 半径
    # =====================================================================
    # 检测先验：从 ROI 框尺寸估算（不再依赖 Z + fx）
    expected_radius = estimate_expected_radius_from_roi(init_box)
    print(f"ROI 估算像素半径: {expected_radius:.2f}")

    init_center_x, init_center_y, init_radius = detect_shaft_center_in_roi(
        first_frame,
        init_box,
        expected_radius_pixels=expected_radius,
        circularity_threshold=config.CIRCULARITY_THRESHOLD,
        radius_tolerance=config.RADIUS_TOLERANCE,
        min_contour_area=config.MIN_CONTOUR_AREA,
        canny_low=config.CANNY_LOW,
        canny_high=config.CANNY_HIGH,
        blur_kernel=config.GAUSSIAN_BLUR_KERNEL,
        return_radius=True
    )
    if init_radius > 0:
        expected_radius = init_radius  # 自适应：用检测半径作后续先验
        print(f"首帧检测半径: {init_radius:.2f} px")

    # =====================================================================
    # 4. 初始化 STARK-ST 跟踪器
    # =====================================================================
    print("[LAUNCH] 加载 STARK-ST 跟踪器...")
    try:
        tracker = StarkTracker(
            checkpoint_rel_path=config.CHECKPOINT_REL_PATH,
            params_name=config.CHECKPOINT_PARAM_NAME,
            dataset_name=config.CHECKPOINT_DATASET_NAME
        )
        tracker.initialize(first_frame, init_box)
        print("[OK] 跟踪器初始化成功！")
    except Exception as e:
        print(f"[ERROR] 跟踪器初始化失败: {e}")
        print("   请确认：")
        print("   1. STARK 代码库路径正确")
        print("   2. 预训练权重已下载到正确位置")
        print("   3. PyTorch + CUDA 环境可用")
        return

    # =====================================================================
    # 5. 逐帧跟踪 + 轮廓检测（存原始像素，延迟换算）
    # =====================================================================
    cv2.namedWindow(config.WINDOW_TRACKING, cv2.WINDOW_NORMAL)
    results = []  # 每项: [frame_idx, pixel_x, pixel_y, radius]

    # 首帧
    results.append([1, init_center_x, init_center_y, init_radius])

    for idx, img_path in enumerate(image_paths[1:], start=2):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN]️ 无法读取图像: {img_path}")
            continue

        # STARK 跟踪
        state = tracker.track(frame)
        pred_box = state['target_bbox']

        # 在跟踪 ROI 内精确检测中心 & 半径
        center_x, center_y, radius = detect_shaft_center_in_roi(
            frame,
            pred_box,
            expected_radius_pixels=expected_radius,
            circularity_threshold=config.CIRCULARITY_THRESHOLD,
            radius_tolerance=config.RADIUS_TOLERANCE,
            min_contour_area=config.MIN_CONTOUR_AREA,
            canny_low=config.CANNY_LOW,
            canny_high=config.CANNY_HIGH,
            blur_kernel=config.GAUSSIAN_BLUR_KERNEL,
            return_radius=True
        )
        # 自适应先验：检测成功则更新
        if radius > 0:
            expected_radius = radius

        print(
            f"Frame {idx}: Pixel=({center_x:.2f}, {center_y:.2f}) "
            f" 半径={radius:.1f} px"
        )

        # 可视化
        x, y, w, h = map(int, pred_box)
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            config.COLOR_BBOX, 2
        )
        cv2.putText(
            frame, f"Frame: {idx}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_TEXT, 2
        )
        cv2.circle(
            frame, (int(center_x), int(center_y)),
            3, config.COLOR_CENTER, -1
        )
        cv2.imshow(config.WINDOW_TRACKING, frame)

        results.append([idx, center_x, center_y, radius])

        key = cv2.waitKey(config.WAIT_KEY_DELAY)
        if key == config.QUIT_KEY:
            break

    cv2.destroyAllWindows()
    print("[OK] 跟踪完成")

    # =====================================================================
    # 5a. 中位数比例尺标定 + 批量坐标换算
    # =====================================================================
    radii = np.array([r[3] for r in results])
    valid = radii[radii > 0]
    N_calib = min(config.SCALE_CALIB_FRAMES, len(valid))
    calib = valid[:N_calib] if len(valid) >= N_calib else valid
    if len(calib) == 0:
        print("[ERROR] 所有帧均未检测到圆，无法标定比例尺")
        return

    median_r = float(np.median(calib))
    scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, median_r)
    print(
        f"[CALIB] 比例尺标定: 有效帧={len(valid)}/{len(results)}, "
        f"前{N_calib}帧半径中位数={median_r:.1f} px → "
        f"scale={scale*1e6:.2f} μm/px "
        f"(= {scale*1e3:.4f} mm/px)"
    )
    print(f"        (Z≈{(median_r * scale / ((config.CX+config.CY)/2) * 1000):.1f} mm 量级, "
          f"但 Z 不参与换算 — 轴径 {config.SHAFT_DIAMETER_MM:.0f} mm 是唯一先验)")

    # 批量换算
    for row in results:
        cam_x, cam_y = pixel_to_mm(row[1], row[2], config.CX, config.CY, scale)
        row.append(cam_x)
        row.append(cam_y)

    # =====================================================================
    # 6. 保存 Excel + 振动幅度计算
    # =====================================================================
    df = pd.DataFrame(
        results,
        columns=["Frame", "Pixel_X", "Pixel_Y", "Radius (px)", "Cam_X (m)", "Cam_Y (m)"]
    )
    output_path = config.OUTPUT_EXCEL
    df.to_excel(output_path, index=False)
    print(f"[OK] 结果已保存到: {output_path}")

    # 振动分析
    cam_x_arr = df["Cam_X (m)"].values
    cam_y_arr = df["Cam_Y (m)"].values
    vibration = compute_vibration_amplitude(cam_x_arr, cam_y_arr)
    print_vibration_result(vibration)

    # 将振幅追加到 Excel
    amplitude_row = pd.DataFrame(
        [[
            "Amplitude", "-", "-", "-",
            vibration['amplitude_x'], vibration['amplitude_y']
        ]],
        columns=df.columns
    )
    df = pd.concat([df, amplitude_row], ignore_index=True)
    df.to_excel(output_path, index=False)

    return df


if __name__ == "__main__":
    main()
