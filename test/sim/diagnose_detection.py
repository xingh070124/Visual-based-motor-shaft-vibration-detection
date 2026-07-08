"""诊断脚本：检查单张仿真图的椭圆检测过程。"""
import os, sys, csv, math
import numpy as np
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
from src import config

F = config.FX_CALIB_NEW
CX = config.CX_CALIB_NEW
CY = config.CY_CALIB_NEW
Z0 = 0.25
D = config.SHAFT_DIAMETER_M
R = D / 2
expected_radius = (D / 2 * F) / Z0

def diagnose(img_path, gt_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {img_path}")
        return

    # 读 GT
    gt = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try: gt[row[0]] = float(row[1])
                except: gt[row[0]] = row[1]

    print(f"图像: {os.path.basename(img_path)}")
    print(f"  GT: theta_true={gt['theta_true_deg']}°, a_gt={gt['ellipse_a_px']:.2f}, b_gt={gt['ellipse_b_px']:.2f}")
    print(f"  expected_radius={expected_radius:.2f}px")

    # ROI
    roi_size = int(2.5 * expected_radius)
    rx, ry = int(CX - roi_size/2), int(CY - roi_size/2)
    rw, rh = roi_size, roi_size
    roi = img[ry:ry+rh, rx:rx+rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu
    otsu_thresh, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  Otsu threshold={otsu_thresh:.1f}")

    contours_otsu, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"  Otsu contours: {len(contours_otsu)}")
    if contours_otsu:
        areas = sorted([cv2.contourArea(c) for c in contours_otsu], reverse=True)[:5]
        print(f"  Top 5 areas: {areas}")

    # CLAHE + Canny
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)
    edges = cv2.Canny(gray_eq, 50, 150)
    contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"  Canny contours: {len(contours_canny)}")
    if contours_canny:
        areas = sorted([cv2.contourArea(c) for c in contours_canny], reverse=True)[:5]
        print(f"  Top 5 areas: {areas}")

    # 尝试直接二值化（轴端面灰度 180，背景 30）
    _, binary_fixed = cv2.threshold(gray_blur, 100, 255, cv2.THRESH_BINARY)
    contours_fixed, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"  Fixed-thresh(100) contours: {len(contours_fixed)}")
    if contours_fixed:
        areas = sorted([cv2.contourArea(c) for c in contours_fixed], reverse=True)[:5]
        print(f"  Top 5 areas: {areas}")
        best = max(contours_fixed, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if len(best) >= 5:
            (cx_r, cy_r), (w, h), ang = cv2.fitEllipse(best)
            a_fit, b_fit = max(w, h)/2, min(w, h)/2
            print(f"  fitEllipse: center=({cx_r+rx:.1f},{cy_r+ry:.1f}), a={a_fit:.2f}, b={b_fit:.2f}, angle={ang:.1f}°")
            theta_est = math.degrees(math.acos(b_fit/a_fit)) if a_fit > 0 else 0
            print(f"  theta_est={theta_est:.4f}° (true={gt['theta_true_deg']}°)")

    # 保存调试图
    debug_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(debug_dir, f'{base}_gray.png'), gray)
    cv2.imwrite(os.path.join(debug_dir, f'{base}_otsu.png'), binary)
    cv2.imwrite(os.path.join(debug_dir, f'{base}_fixed.png'), binary_fixed)
    cv2.imwrite(os.path.join(debug_dir, f'{base}_edges.png'), edges)
    print(f"  调试图保存到: {debug_dir}/{base}_*.png")


if __name__ == "__main__":
    images_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'images')
    gt_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'ground_truth')

    for name in ['theta00_phase000_s00_b00_r1.png', 'theta10_phase000_s00_b00_r1.png',
                  'theta15_phase000_s00_b00_r1.png', 'theta20_phase000_s00_b00_r1.png']:
        img_path = os.path.join(images_dir, name)
        gt_path = os.path.join(gt_dir, name.replace('.png', '.csv'))
        if os.path.exists(img_path):
            print(f"\n{'='*60}")
            diagnose(img_path, gt_path)
