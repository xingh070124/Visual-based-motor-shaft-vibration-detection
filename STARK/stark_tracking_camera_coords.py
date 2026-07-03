#stark_tracking_camera_coords9733.xlsx
import os
import cv2
import numpy as np
import pandas as pd
import re
import glob
from lib.test.evaluation.tracker import Tracker
from lib.test.parameter.stark_st import parameters

# 二倍图像  六月份
# fx = 1791.22202940993
# fy = 2397.85179602954
# cx = 230.701118313539
# cy = 185.173943174303
# K = np.array([[fx, 2.18133573006278, cx], [0, fy, cy], [0, 0, 1]])
# K_inv = np.linalg.inv(K)
# Z = 0.1582222  # 深度（米）


# ====== Camera Parameters ======    一倍
fx, fy = 871.634556391140, 1159.30440189264
cx, cy = 375.114519279214, 210.871016530132
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_inv = np.linalg.inv(K)
Z = 0.1817  # Depth (m)

# 电机轴直径（米）
shaft_diameter_m = 0.012  # 12mm = 0.012m

# 计算预期像素半径（近似，使用fx和Z）
expected_radius_pixels = (shaft_diameter_m / 2 * fx) / Z
print(f"Expected radius in pixels: {expected_radius_pixels:.2f}")


# 图像排序函数（自然顺序）
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def detect_shaft_center_in_roi(frame, roi_box):
    """
    在给定的ROI中进行轮廓提取，检测电机轴的圆形轮廓，并返回圆心的坐标（相对于原图像）。
    如果未检测到合适的轮廓，回退到ROI中心。
    """
    x, y, w, h = map(int, roi_box)
    roi = frame[y:y + h, x:x + w]

    # 预处理：灰度化 + 高斯模糊
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours detected in ROI. Falling back to bbox center.")
        return x + w / 2, y + h / 2

    best_contour = None
    best_circularity = 0

    # 过滤轮廓：选择最圆的（圆度接近1）
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 忽略太小的轮廓
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = cnt

    if best_contour is not None and best_circularity > 0.5:  # 阈值，确保足够圆
        # 使用最小包围圆获取中心和半径
        (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(best_contour)
        # 检查半径是否接近预期
        if abs(radius - expected_radius_pixels) < expected_radius_pixels * 0.3:  # 允许30%偏差
            center_x = x + cx_roi
            center_y = y + cy_roi
            print(
                f"Detected shaft center: ({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}, circularity={best_circularity:.2f}")
            return center_x, center_y

    print("No suitable circular contour detected. Falling back to bbox center.")
    return x + w / 2, y + h / 2


def main():
    # 1. 图像路径设置
    image_folder = "frames200-1"
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")), key=natural_key)  # 使用自然排序

    if not image_paths:
        print("❌ No images found in the directory!")
        return

    # 2. 加载首帧 & 选择初始 ROI
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print("❌ Failed to read the first image.")
        return

    cv2.namedWindow('Select Target', cv2.WINDOW_KEEPRATIO)
    init_box = cv2.selectROI("Select Target", first_frame, False, False)
    cv2.destroyWindow("Select Target")
    if sum(init_box) == 0:
        print("❌ Target selection cancelled.")
        return
    init_box = [int(v) for v in init_box]

    # 在初始ROI中进行轮廓提取以精确初始中心
    init_center_x, init_center_y = detect_shaft_center_in_roi(first_frame, init_box)

    # 3. 初始化 STARK-ST 跟踪器
    print("✅ Loading STARK-ST tracker...")
    try:
        params = parameters("baseline")
        params.checkpoint = 'checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth'
        tracker = Tracker(name='stark_st2', parameter_name='baseline', dataset_name='OTB100').create_tracker(params)
        print("✅ Tracker initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to create tracker: {e}")
        return

    # 初始化跟踪器（使用原始init_box，因为跟踪器需要bbox）
    tracker.initialize(first_frame, {'init_bbox': init_box})

    # 4. 跟踪与转换记录
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    results = []
    # 为首帧添加结果
    pixel = np.array([init_center_x, init_center_y, 1.0])
    cam_coord = Z * (K_inv @ pixel)
    X, Y, _ = cam_coord
    results.append([1, init_center_x, init_center_y, X, Y])
    print(f"Frame 1: Pixel Center = ({init_center_x:.2f}, {init_center_y:.2f}) --> Camera Coord = ({X:.4f}, {Y:.4f})")

    for idx, img_path in enumerate(image_paths[1:], start=2):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Failed to read image: {img_path}")
            continue

        state = tracker.track(frame)
        pred_box = state['target_bbox']

        # 在跟踪的ROI中进行轮廓提取以精确中心
        center_x, center_y = detect_shaft_center_in_roi(frame, pred_box)

        # 像素坐标 → 相机坐标
        pixel = np.array([center_x, center_y, 1.0])
        cam_coord = Z * (K_inv @ pixel)
        X, Y, _ = cam_coord

        print(f"Frame {idx}: Pixel Center = ({center_x:.2f}, {center_y:.2f}) --> Camera Coord = ({X:.4f}, {Y:.4f})")

        # 可视化
        x, y, w, h = map(int, pred_box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
        cv2.imshow("Tracking", frame)

        results.append([idx, center_x, center_y, X, Y])

        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyAllWindows()
    print("✅ Tracking finished.")

    # 5. 保存为 Excel
    df = pd.DataFrame(results, columns=["Frame", "Pixel_X", "Pixel_Y", "Cam_X (m)", "Cam_Y (m)"])
    df.to_excel("stark_tracking_camera_coords200-1.xlsx", index=False)
    print("✅ Results saved to Excel: stark_tracking_camera_coords200-1.xlsx")

    # 6. 计算振动幅度（X和Y方向的最大振幅，以及总幅度）
    cam_x = df["Cam_X (m)"].values
    cam_y = df["Cam_Y (m)"].values
    amplitude_x = np.max(cam_x) - np.min(cam_x)
    amplitude_y = np.max(cam_y) - np.min(cam_y)
    amplitude_total = np.sqrt(amplitude_x ** 2 + amplitude_y ** 2)  # 总幅度（欧氏距离）

    print(f"振动幅度 (X方向): {amplitude_x * 1000:.2f} mm")
    print(f"振动幅度 (Y方向): {amplitude_y * 1000:.2f} mm")
    print(f"总振动幅度: {amplitude_total * 1000:.2f} mm")

    # 可选：将幅度添加到Excel（作为新行）
    amplitude_row = pd.DataFrame([["Amplitude", "-", "-", amplitude_x, amplitude_y]], columns=df.columns)
    df = pd.concat([df, amplitude_row], ignore_index=True)
    df.to_excel("stark_tracking_camera_coords200-1.xlsx", index=False)


if __name__ == "__main__":
    main()



