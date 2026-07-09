"""Phase 8: 分离 scale 误差和中心检测误差评估。

问题背景：
  评估报告显示 θ=20° 时各向异性 vs naive 仅差 0.42%，理论预测应为 3.16%。
  怀疑中心检测噪声（~0.1px → 1.7%）淹没了 scale 改善。

方法：
  1. 用 GT 中心坐标做坐标换算（消除中心噪声），单独验证 scale 理论改善量
  2. 用检测中心坐标做坐标换算，对比中心噪声的影响
  3. 检查仿真渲染的实际倾角是否与标称值一致

用法：
    D:\\anaconda\\python.exe test/sim/evaluate_scale_vs_center.py
"""

import os
import sys
import csv
import json
import math
import numpy as np
import cv2
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.tracking.shaft_detector import detect_ellipse_in_roi
from src.tracking.coordinate import (
    estimate_tilt_angle,
    pixel_to_mm_anisotropic,
    pixel_to_mm,
    compute_scale_m_per_px,
)

F = config.FX_CALIB_NEW
CX = config.CX_CALIB_NEW
CY = config.CY_CALIB_NEW
Z0 = 0.25
D = config.SHAFT_DIAMETER_M


def load_gt(gt_path):
    """读取 GT CSV。"""
    gt = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    gt[row[0]] = float(row[1])
                except (ValueError, TypeError):
                    gt[row[0]] = row[1]
    return gt


def evaluate_single(img_path, gt, expected_radius):
    """对单张图做分离评估。"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    roi_size = int(2.5 * expected_radius)
    roi_box = (CX - roi_size / 2, CY - roi_size / 2, roi_size, roi_size)

    # === 检测椭圆 ===
    det_cx, det_cy, det_a, det_b, det_angle = detect_ellipse_in_roi(
        img, roi_box, expected_radius, return_params=True
    )

    # === GT 参数 ===
    gt_cx = gt.get('ellipse_cx_px', CX)
    gt_cy = gt.get('ellipse_cy_px', CY)
    gt_a = gt.get('ellipse_a_px', expected_radius)
    gt_b = gt.get('ellipse_b_px', expected_radius)
    gt_angle = gt.get('ellipse_angle_deg', 0.0)

    theta_true = gt['theta_true_deg']
    X_true = gt['vibration_x_m']
    Y_true = gt['vibration_y_m']
    amp_true = math.sqrt(X_true ** 2 + Y_true ** 2)

    # === 方案1: GT 中心 + 检测 scale（各向异性） ===
    scale_major_det = compute_scale_m_per_px(D, det_a)
    scale_minor_det = compute_scale_m_per_px(D, det_b)
    X1, Y1 = pixel_to_mm_anisotropic(gt_cx, gt_cy, CX, CY,
                                      scale_major_det, scale_minor_det, det_angle)
    amp1 = math.sqrt(X1 ** 2 + Y1 ** 2)
    err1 = abs(amp1 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 方案2: GT 中心 + 检测 scale（naive 各向同性） ===
    naive_scale_det = compute_scale_m_per_px(D, math.sqrt(det_a * det_b))
    X2, Y2 = pixel_to_mm(gt_cx, gt_cy, CX, CY, naive_scale_det)
    amp2 = math.sqrt(X2 ** 2 + Y2 ** 2)
    err2 = abs(amp2 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 方案3: GT 中心 + GT scale（各向异性） ===
    scale_major_gt = compute_scale_m_per_px(D, gt_a)
    scale_minor_gt = compute_scale_m_per_px(D, gt_b)
    X3, Y3 = pixel_to_mm_anisotropic(gt_cx, gt_cy, CX, CY,
                                      scale_major_gt, scale_minor_gt, gt_angle)
    amp3 = math.sqrt(X3 ** 2 + Y3 ** 2)
    err3 = abs(amp3 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 方案4: GT 中心 + GT scale（naive 各向同性） ===
    naive_scale_gt = compute_scale_m_per_px(D, math.sqrt(gt_a * gt_b))
    X4, Y4 = pixel_to_mm(gt_cx, gt_cy, CX, CY, naive_scale_gt)
    amp4 = math.sqrt(X4 ** 2 + Y4 ** 2)
    err4 = abs(amp4 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 方案5: 检测中心 + 检测 scale（各向异性）— 原始管线 ===
    X5, Y5 = pixel_to_mm_anisotropic(det_cx, det_cy, CX, CY,
                                      scale_major_det, scale_minor_det, det_angle)
    amp5 = math.sqrt(X5 ** 2 + Y5 ** 2)
    err5 = abs(amp5 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 方案6: 检测中心 + 检测 scale（naive）— 原始管线 ===
    X6, Y6 = pixel_to_mm(det_cx, det_cy, CX, CY, naive_scale_det)
    amp6 = math.sqrt(X6 ** 2 + Y6 ** 2)
    err6 = abs(amp6 - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 中心检测误差 ===
    center_err_px = math.sqrt((det_cx - gt_cx) ** 2 + (det_cy - gt_cy) ** 2)

    # === scale 检测误差 ===
    a_err_pct = abs(det_a - gt_a) / gt_a * 100 if gt_a > 0 else 0.0
    b_err_pct = abs(det_b - gt_b) / gt_b * 100 if gt_b > 0 else 0.0

    # === 理论 scale 改善量 ===
    # naive scale 在 θ 方向的误差 = 1/sqrt(cos θ) - 1
    if theta_true > 0.01:
        theoretical_naive_err = abs(1.0 / math.sqrt(math.cos(math.radians(theta_true))) - 1.0) * 100
    else:
        theoretical_naive_err = 0.0

    return {
        'image': os.path.basename(img_path),
        'theta_true': theta_true,
        # GT 参数
        'gt_a': gt_a,
        'gt_b': gt_b,
        'gt_cx': gt_cx,
        'gt_cy': gt_cy,
        'gt_angle': gt_angle,
        # 检测参数
        'det_a': det_a,
        'det_b': det_b,
        'det_cx': det_cx,
        'det_cy': det_cy,
        'det_angle': det_angle,
        # 检测误差
        'center_err_px': center_err_px,
        'a_err_pct': a_err_pct,
        'b_err_pct': b_err_pct,
        # 振幅误差（6种方案）
        'err_gt_center_aniso': err1 * 100,
        'err_gt_center_naive': err2 * 100,
        'err_gt_center_gt_scale_aniso': err3 * 100,
        'err_gt_center_gt_scale_naive': err4 * 100,
        'err_det_center_aniso': err5 * 100,
        'err_det_center_naive': err6 * 100,
        # 理论值
        'theoretical_naive_err_pct': theoretical_naive_err,
        # 其他
        'sigma': gt.get('sigma', 0),
        'blur_len': gt.get('blur_len', 0),
        'amp_true_um': amp_true * 1e6,
    }


def main():
    images_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'images')
    gt_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'ground_truth')
    results_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'results')
    os.makedirs(results_dir, exist_ok=True)

    expected_radius = (D / 2 * F) / Z0
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    print(f"分离评估: {len(image_files)} 张图")
    print(f"  预期半径: {expected_radius:.2f}px")

    rows = []
    for i, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        gt_path = os.path.join(gt_dir, fname.replace('.png', '.csv'))
        if not os.path.exists(gt_path):
            continue
        gt = load_gt(gt_path)
        result = evaluate_single(img_path, gt, expected_radius)
        if result:
            rows.append(result)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(image_files)}]")

    if not rows:
        print("[ERROR] 无有效结果")
        return 1

    # 保存逐图结果
    out_csv = os.path.join(results_dir, 'scale_vs_center_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n逐图结果: {out_csv}")

    # === 按倾角汇总 ===
    by_theta = defaultdict(list)
    for r in rows:
        by_theta[r['theta_true']].append(r)

    print(f"\n{'='*100}")
    print("分离评估汇总 — 按 θ 分组")
    print(f"{'='*100}")
    print(f"{'θ(°)':>5} | {'GT中心+检测scale':>20} | {'GT中心+GTscale':>20} | "
          f"{'检测中心+检测scale':>20} | {'理论naive误差':>14} | {'中心检测误差':>14}")
    print(f"{'':>5} | {'各向异性 / naive':>20} | {'各向异性 / naive':>20} | "
          f"{'各向异性 / naive':>20} | {'(%)':>14} | {'(px)':>14}")
    print("-" * 100)

    summary = []
    for theta in sorted(by_theta.keys()):
        rs = by_theta[theta]
        # GT中心 + 检测scale
        gt_aniso = np.mean([r['err_gt_center_aniso'] for r in rs])
        gt_naive = np.mean([r['err_gt_center_naive'] for r in rs])
        # GT中心 + GT scale
        gt_gt_aniso = np.mean([r['err_gt_center_gt_scale_aniso'] for r in rs])
        gt_gt_naive = np.mean([r['err_gt_center_gt_scale_naive'] for r in rs])
        # 检测中心 + 检测scale
        det_aniso = np.mean([r['err_det_center_aniso'] for r in rs])
        det_naive = np.mean([r['err_det_center_naive'] for r in rs])
        # 理论
        theo = np.mean([r['theoretical_naive_err_pct'] for r in rs])
        # 中心检测误差
        center_err = np.mean([r['center_err_px'] for r in rs])

        # scale 改善量 = naive - 各向异性
        gt_scale_improvement = gt_naive - gt_aniso
        gt_gt_scale_improvement = gt_gt_naive - gt_gt_aniso
        det_scale_improvement = det_naive - det_aniso

        print(f"{theta:5.0f} | {gt_aniso:8.3f}% / {gt_naive:8.3f}% | "
              f"{gt_gt_aniso:8.3f}% / {gt_gt_naive:8.3f}% | "
              f"{det_aniso:8.3f}% / {det_naive:8.3f}% | "
              f"{theo:12.3f}% | {center_err:12.4f}")

        summary.append({
            'theta_true': theta,
            'count': len(rs),
            'gt_center_aniso_err_pct': gt_aniso,
            'gt_center_naive_err_pct': gt_naive,
            'gt_center_scale_improvement_pct': gt_scale_improvement,
            'gt_center_gt_scale_aniso_err_pct': gt_gt_aniso,
            'gt_center_gt_scale_naive_err_pct': gt_gt_naive,
            'gt_center_gt_scale_improvement_pct': gt_gt_scale_improvement,
            'det_center_aniso_err_pct': det_aniso,
            'det_center_naive_err_pct': det_naive,
            'det_center_scale_improvement_pct': det_scale_improvement,
            'theoretical_naive_err_pct': theo,
            'center_detection_err_px': center_err,
            'a_detection_err_pct': np.mean([r['a_err_pct'] for r in rs]),
            'b_detection_err_pct': np.mean([r['b_err_pct'] for r in rs]),
        })

    print(f"\n{'='*100}")
    print("结论:")
    for s in summary:
        t = s['theta_true']
        print(f"  θ={t:.0f}°: "
              f"GT中心+GTscale 改善={s['gt_center_gt_scale_improvement_pct']:.3f}% (理论={s['theoretical_naive_err_pct']:.3f}%), "
              f"检测中心+检测scale 改善={s['det_center_scale_improvement_pct']:.3f}%, "
              f"中心误差={s['center_detection_err_px']:.4f}px")

    # 保存汇总
    summary_csv = os.path.join(results_dir, 'scale_vs_center_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n汇总: {summary_csv}")

    # JSON 报告
    report = {
        'n_images': len(rows),
        'summary': summary,
    }
    report_path = os.path.join(results_dir, 'scale_vs_center_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
