"""Phase 4+6: 反演精度评估脚本。

对 640 张仿真静态图跑椭圆检测 + 倾角反演 + 各向异性坐标换算，
对比 ground truth，输出：
  1. results/per_image_results.csv — 每张图的逐项结果
  2. results/summary_by_theta.csv — 按倾角分组的统计
  3. results/summary_by_noise.csv — 按噪声水平分组的统计
  4. results/eval_report.json — 验收标准核对
  5. figures/theta_error_vs_noise.png — 倾角误差-噪声曲线
  6. figures/amplitude_error_vs_theta.png — 振幅误差-倾角曲线

用法：
    D:\\anaconda\\python.exe test/sim/evaluate_tilt_correction.py
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import numpy as np
import cv2
from collections import defaultdict

# 项目根目录
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


# =============================================================================
# 物理参数
# =============================================================================

F = config.FX_CALIB_NEW
CX = config.CX_CALIB_NEW
CY = config.CY_CALIB_NEW
Z0 = 0.25
D = config.SHAFT_DIAMETER_M
IMG_W, IMG_H = 1920, 1080

# 仿真参数
AMP_UM = 500.0  # 与渲染一致
AMP_M = AMP_UM * 1e-6


# =============================================================================
# 单图评估
# =============================================================================

def evaluate_image(img_path, gt_dict, expected_radius, use_anisotropic=True):
    """对单张图跑椭圆检测 + 反演 + 坐标换算，返回评估结果字典。

    Args:
        img_path: 图像路径
        gt_dict: 从 GT CSV 读出的字典
        expected_radius: 预期像素半径
        use_anisotropic: 是否使用各向异性 scale

    Returns:
        评估结果字典
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    # ROI 假设在中心，覆盖 2*expected_radius 范围
    roi_size = int(2.5 * expected_radius)
    roi_box = (
        CX - roi_size / 2,
        CY - roi_size / 2,
        roi_size,
        roi_size
    )

    # 椭圆检测
    cx, cy, a, b, angle = detect_ellipse_in_roi(
        img, roi_box, expected_radius,
        return_params=True
    )

    # 反演倾角
    theta_est = estimate_tilt_angle(a, b)

    # 各向异性 scale
    scale_major = compute_scale_m_per_px(D, a)
    scale_minor = compute_scale_m_per_px(D, b)
    naive_scale = compute_scale_m_per_px(D, math.sqrt(a * b))

    # 真实振动（米）
    X_true = gt_dict['vibration_x_m']
    Y_true = gt_dict['vibration_y_m']
    theta_true = gt_dict['theta_true_deg']
    amp_true = math.sqrt(X_true ** 2 + Y_true ** 2)

    # === 振幅评估（用检测到的 a/b 推算的 scale） ===
    if use_anisotropic:
        X_est, Y_est = pixel_to_mm_anisotropic(
            cx, cy, CX, CY,
            scale_major, scale_minor, angle
        )
    else:
        X_est, Y_est = pixel_to_mm(cx, cy, CX, CY, naive_scale)

    amp_meas = math.sqrt(X_est ** 2 + Y_est ** 2)
    amp_rel_err = abs(amp_meas - amp_true) / amp_true if amp_true > 0 else 0.0

    # === 振幅评估（用理论 a/b 推算的 scale —— 验证检测精度对反演的影响） ===
    a_ideal = (D / 2 * F) / (Z0 * math.cos(theta_true * math.pi / 180))
    b_ideal = (D / 2 * F) / Z0
    scale_major_ideal = D / (2 * a_ideal)
    scale_minor_ideal = D / (2 * b_ideal)
    naive_scale_ideal = D / (2 * math.sqrt(a_ideal * b_ideal))
    if use_anisotropic:
        X_ideal, Y_ideal = pixel_to_mm_anisotropic(
            cx, cy, CX, CY,
            scale_major_ideal, scale_minor_ideal, 0  # 假设长轴沿 u 方向
        )
    else:
        X_ideal, Y_ideal = pixel_to_mm(cx, cy, CX, CY, naive_scale_ideal)
    amp_ideal_meas = math.sqrt(X_ideal ** 2 + Y_ideal ** 2)
    amp_rel_err_ideal_scale = abs(amp_ideal_meas - amp_true) / amp_true if amp_true > 0 else 0.0

    # 倾角误差
    theta_err = abs(theta_est - theta_true)

    return {
        'image': os.path.basename(img_path),
        'theta_true': theta_true,
        'theta_est': theta_est,
        'theta_err': theta_err,
        'ellipse_a': a,
        'ellipse_b': b,
        'ellipse_angle': angle,
        'center_x': cx,
        'center_y': cy,
        'scale_major': scale_major,
        'scale_minor': scale_minor,
        'naive_scale': naive_scale,
        'X_est_m': X_est,
        'Y_est_m': Y_est,
        'X_true_m': X_true,
        'Y_true_m': Y_true,
        'amp_meas_m': amp_meas,
        'amp_true_m': amp_true,
        'amp_rel_err': amp_rel_err,
        'amp_rel_err_ideal_scale': amp_rel_err_ideal_scale,
        'sigma': gt_dict['sigma'],
        'blur_len': gt_dict['blur_len'],
        'vibration_phase': gt_dict['vibration_phase_deg'],
    }


# =============================================================================
# 批量评估
# =============================================================================

def load_ground_truth(gt_path):
    """读取单个 GT CSV。"""
    d = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳表头
        for row in reader:
            if len(row) >= 2:
                key, val = row[0], row[1]
                try:
                    d[key] = float(val)
                except (ValueError, TypeError):
                    d[key] = val
    return d


def batch_evaluate(images_dir, gt_dir, results_dir, max_n=None, use_anisotropic=True):
    """批量评估所有仿真图。"""
    # 预期半径（θ=0° 时）
    expected_radius = (D / 2 * F) / Z0

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    if max_n:
        image_files = image_files[:max_n]

    print(f"开始评估 {len(image_files)} 张图像...")
    print(f"  预期像素半径: {expected_radius:.2f}px")
    print(f"  评估模式: {'各向异性' if use_anisotropic else 'naive (各向同性)'}")

    rows = []
    t0 = time.time()

    for i, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        gt_path = os.path.join(gt_dir, fname.replace('.png', '.csv'))

        if not os.path.exists(gt_path):
            continue

        gt = load_ground_truth(gt_path)
        result = evaluate_image(img_path, gt, expected_radius, use_anisotropic)
        if result:
            rows.append(result)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(image_files) - i - 1) / rate
            print(f"  [{i+1:4d}/{len(image_files)}] "
                  f"{rate:.1f} img/s, ETA {eta:.0f}s")

    # 保存逐图结果
    per_image_csv = os.path.join(results_dir, 'per_image_results.csv')
    with open(per_image_csv, 'w', newline='', encoding='utf-8') as f:
        if not rows:
            print("[WARN] 没有有效结果")
            return []
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n逐图结果: {per_image_csv}")

    return rows


# =============================================================================
# 汇总统计
# =============================================================================

def summary_by_key(rows, key, results_dir):
    """按某列分组统计。"""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)

    summary = []
    for k in sorted(groups.keys()):
        rs = groups[k]
        theta_errs = [r['theta_err'] for r in rs]
        amp_errs = [r['amp_rel_err'] for r in rs]
        summary.append({
            key: k,
            'count': len(rs),
            'theta_err_mean': np.mean(theta_errs),
            'theta_err_std': np.std(theta_errs),
            'theta_err_max': np.max(theta_errs),
            'amp_rel_err_mean': np.mean(amp_errs) * 100,  # 百分比
            'amp_rel_err_std': np.std(amp_errs) * 100,
            'amp_rel_err_max': np.max(amp_errs) * 100,
        })

    out_csv = os.path.join(results_dir, f'summary_by_{key}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"按 {key} 汇总: {out_csv}")
    return summary


def check_acceptance(rows):
    """对照方案文档的验收标准。"""
    # 按倾角统计 θ_err
    by_theta = defaultdict(list)
    for r in rows:
        by_theta[r['theta_true']].append(r)

    # 验收标准（来自方案文档 Phase 4）
    checks = []

    for theta, rs in sorted(by_theta.items()):
        theta_errs = np.array([r['theta_err'] for r in rs])
        amp_errs = np.array([r['amp_rel_err'] for r in rs])

        if len(rs) == 0:
            continue

        check = {
            'theta_true': theta,
            'count': len(rs),
            'theta_err_mean': float(np.mean(theta_errs)),
            'theta_err_max': float(np.max(theta_errs)),
            'amp_rel_err_mean_pct': float(np.mean(amp_errs) * 100),
            'amp_rel_err_max_pct': float(np.max(amp_errs) * 100),
        }

        # Phase 4 验收
        # sigma<=3 水平下，theta_err < 1°
        # sigma=8 水平下，theta_err < 2°
        # sigma=15 水平下，theta_err < 3°
        low_noise = [r for r in rs if r['sigma'] <= 3]
        med_noise = [r for r in rs if r['sigma'] <= 8]
        all_noise = rs

        if low_noise:
            check['theta_err_low_noise_max'] = float(np.max([r['theta_err'] for r in low_noise]))
        if med_noise:
            check['theta_err_med_noise_max'] = float(np.max([r['theta_err'] for r in med_noise]))
        check['theta_err_all_max'] = float(np.max(theta_errs))

        # Phase 5 验收：振幅误差 < 1%
        # 但只在无噪声模糊的"理想"情况下
        ideal = [r for r in rs if r['sigma'] == 0 and r['blur_len'] == 0]
        if ideal:
            check['amp_rel_err_ideal_max_pct'] = float(np.max([r['amp_rel_err'] for r in ideal]) * 100)
            check['amp_rel_err_ideal_mean_pct'] = float(np.mean([r['amp_rel_err'] for r in ideal]) * 100)

        checks.append(check)

    return checks


# =============================================================================
# 绘图
# =============================================================================

def plot_theta_error_vs_noise(rows, out_path):
    """倾角误差-噪声曲线（按倾角分组，X 轴为噪声水平）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 按 (theta, sigma) 聚合
    by_ts = defaultdict(list)
    for r in rows:
        by_ts[(r['theta_true'], r['sigma'])].append(r['theta_err'])

    thetas = sorted(set(r['theta_true'] for r in rows))
    sigmas = sorted(set(r['sigma'] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    for theta in thetas:
        ys = []
        for sigma in sigmas:
            errs = by_ts.get((theta, sigma), [])
            ys.append(np.mean(errs) if errs else 0)
        ax.plot(sigmas, ys, marker='o', label=f'θ={theta}°')

    ax.set_xlabel('Noise σ (灰度)')
    ax.set_ylabel('Tilt Angle Error (°)')
    ax.set_title('倾角反演精度 vs 噪声水平（按倾角分组）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"图: {out_path}")


def plot_amplitude_error_vs_theta(rows, out_path):
    """振幅误差-倾角曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 按 (theta, noise) 聚合
    by_tn = defaultdict(list)
    for r in rows:
        by_tn[(r['theta_true'], r['sigma'])].append(r['amp_rel_err'])

    thetas = sorted(set(r['theta_true'] for r in rows))
    sigmas = sorted(set(r['sigma'] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    for sigma in sigmas:
        ys = []
        for theta in thetas:
            errs = by_tn.get((theta, sigma), [])
            ys.append(np.mean(errs) * 100 if errs else 0)
        ax.plot(thetas, ys, marker='s', label=f'σ={sigma}')

    ax.set_xlabel('Tilt Angle (°)')
    ax.set_ylabel('Amplitude Relative Error (%)')
    ax.set_title('振幅相对误差 vs 倾角（各向异性 scale 补偿后）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"图: {out_path}")


def plot_anisotropy_ratio(rows, out_path):
    """各向异性比 |AmpX / AmpY| 随倾角变化。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 收集 (theta, ratio)
    by_theta = defaultdict(list)
    for r in rows:
        if abs(r['Y_est_m']) > 1e-9:
            ratio = abs(r['X_est_m'] / r['Y_est_m'])
        else:
            ratio = float('inf')
        by_theta[r['theta_true']].append(ratio)

    thetas = sorted(by_theta.keys())
    means = [np.mean([r for r in by_theta[t] if np.isfinite(r)]) for t in thetas]
    stds = [np.std([r for r in by_theta[t] if np.isfinite(r)]) for t in thetas]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(thetas, means, yerr=stds, marker='o', capsize=4)
    ax.axhline(1.0, color='r', linestyle='--', label='理想各向同性 (1.0)')
    ax.set_xlabel('Tilt Angle (°)')
    ax.set_ylabel('|X_est / Y_est|')
    ax.set_title('各向异性比 — 补偿后应接近 1.0')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"图: {out_path}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4+6: 倾斜补偿反演精度评估'
    )
    parser.add_argument('--images-dir', default='test/sim/images')
    parser.add_argument('--gt-dir', default='test/sim/ground_truth')
    parser.add_argument('--results-dir', default='test/sim/results')
    parser.add_argument('--figures-dir', default='test/sim/figures')
    parser.add_argument('--max-n', type=int, default=None,
                        help='限制评估图像数（用于快速测试）')
    parser.add_argument('--mode', choices=['anisotropic', 'naive', 'both'],
                        default='anisotropic',
                        help='评估模式: anisotropic（各向异性）/ naive（各向同性）/ both（都跑）')
    args = parser.parse_args()

    images_dir = os.path.join(_PROJECT_ROOT, args.images_dir)
    gt_dir = os.path.join(_PROJECT_ROOT, args.gt_dir)
    results_dir = os.path.join(_PROJECT_ROOT, args.results_dir)
    figures_dir = os.path.join(_PROJECT_ROOT, args.figures_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    modes = ['anisotropic', 'naive'] if args.mode == 'both' else [args.mode]

    all_reports = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"模式: {mode}")
        print(f"{'='*60}")

        rows = batch_evaluate(
            images_dir, gt_dir, results_dir,
            max_n=args.max_n, use_anisotropic=(mode == 'anisotropic')
        )

        if not rows:
            continue

        # 汇总
        summary_theta = summary_by_key(rows, 'theta_true', results_dir)
        summary_sigma = summary_by_key(rows, 'sigma', results_dir)
        summary_blur = summary_by_key(rows, 'blur_len', results_dir)

        # 验收检查
        checks = check_acceptance(rows)
        report = {
            'mode': mode,
            'n_images': len(rows),
            'by_theta': checks,
        }
        all_reports[mode] = report

        # 绘图
        try:
            plot_theta_error_vs_noise(
                rows, os.path.join(figures_dir, f'theta_error_vs_noise_{mode}.png')
            )
            plot_amplitude_error_vs_theta(
                rows, os.path.join(figures_dir, f'amplitude_error_vs_theta_{mode}.png')
            )
            plot_anisotropy_ratio(
                rows, os.path.join(figures_dir, f'anisotropy_ratio_{mode}.png')
            )
        except Exception as e:
            print(f"[WARN] 绘图失败: {e}")

    # 保存 JSON 报告
    report_path = os.path.join(results_dir, 'eval_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {report_path}")

    # 打印核心结论
    print(f"\n{'='*60}")
    print("评估总结")
    print(f"{'='*60}")
    for mode, rep in all_reports.items():
        print(f"\n模式: {mode}")
        print(f"  评估图像数: {rep['n_images']}")
        for chk in rep['by_theta']:
            t = chk['theta_true']
            te_mean = chk['theta_err_mean']
            te_max = chk['theta_err_max']
            ae_mean = chk.get('amp_rel_err_ideal_mean_pct', None)
            ae_max = chk.get('amp_rel_err_ideal_max_pct', None)
            print(f"  θ={t:2.0f}°: θ_err mean={te_mean:.3f}°/max={te_max:.3f}°, "
                  f"amp_err ideal mean={ae_mean:.2f}%" if ae_mean is not None
                  else f"  θ={t:2.0f}°: θ_err mean={te_mean:.3f}°/max={te_max:.3f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
