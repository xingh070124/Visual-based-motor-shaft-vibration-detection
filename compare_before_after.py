# -*- coding: utf-8 -*-
"""修复前后视频测试对比脚本。

读取 test/output/（修复前）与 test/output_fixed/（修复后）的 *_data.xlsx，
计算统一指标集，生成对比报告（Markdown + Excel + 图表）。

用法：
  D:\\anaconda\\python.exe compare_before_after.py
  D:\\anaconda\\python.exe compare_before_after.py --before test/output --after test/output_fixed --exclude 90,100
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sig

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
THEORETICAL_SCALE = 60.0  # 理论 scale μm/px (12000/(2*100))


def natural_key(s):
    """自然排序键函数。"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def find_video_dirs(output_root, exclude):
    """扫描输出目录，返回排序后的视频名列表（排除指定视频）。"""
    if not os.path.isdir(output_root):
        return []
    excl_set = set()
    if exclude:
        excl_set = {x.strip() for x in exclude.split(',') if x.strip()}
    names = []
    for d in os.listdir(output_root):
        full = os.path.join(output_root, d)
        if os.path.isdir(full) and d not in excl_set and d != 'analysis':
            # 确认有 _data.xlsx 文件
            xlsx = os.path.join(full, f"{d}_data.xlsx")
            if os.path.exists(xlsx):
                names.append(d)
    return sorted(names, key=natural_key)


def load_video_metrics(output_root, video_name):
    """读取单个视频的 data.xlsx，计算全部指标，返回 dict。"""
    xlsx_path = os.path.join(output_root, video_name, f"{video_name}_data.xlsx")
    if not os.path.exists(xlsx_path):
        return None

    try:
        raw = pd.read_excel(xlsx_path, sheet_name='Raw Data')
        summary = pd.read_excel(xlsx_path, sheet_name='Summary')
    except Exception as e:
        print(f"  [WARN] 读取 {xlsx_path} 失败: {e}")
        return None

    # 解析 scale
    scale = np.nan
    for _, row in summary.iterrows():
        param = str(row.get('Parameter', ''))
        if 'Scale' in param:
            try:
                scale = float(row['Value'])
            except (ValueError, TypeError):
                pass
            break

    # Radius 统计（仅 radius > 0）
    radius_col = raw['Radius (px)'].values
    valid_r = radius_col[radius_col > 0]
    radius_mean = float(np.mean(valid_r)) if len(valid_r) > 0 else 0.0
    radius_std = float(np.std(valid_r)) if len(valid_r) > 0 else 0.0
    radius_cv = (radius_std / radius_mean * 100) if radius_mean > 0 else 100.0
    detection_rate = (len(valid_r) / len(radius_col) * 100) if len(radius_col) > 0 else 0.0

    # 振幅（原始）
    cam_x = raw['Cam_X (m)'].values
    cam_y = raw['Cam_Y (m)'].values
    amp_x = float(np.max(cam_x) - np.min(cam_x)) * 1000
    amp_y = float(np.max(cam_y) - np.min(cam_y)) * 1000
    amp_total = float(np.sqrt(amp_x ** 2 + amp_y ** 2))

    # 去漂移振幅（scipy.signal.detrend, linear）
    if len(cam_x) >= 3:
        cam_x_dt = sig.detrend(cam_x, type='linear')
        cam_y_dt = sig.detrend(cam_y, type='linear')
    else:
        cam_x_dt = cam_x
        cam_y_dt = cam_y
    amp_x_dt = float(np.max(cam_x_dt) - np.min(cam_x_dt)) * 1000
    amp_y_dt = float(np.max(cam_y_dt) - np.min(cam_y_dt)) * 1000
    amp_total_dt = float(np.sqrt(amp_x_dt ** 2 + amp_y_dt ** 2))

    # 漂移量（后1/3均值 - 前1/3均值）
    n = len(cam_x)
    third = max(1, n // 3)
    drift_x = float(np.mean(cam_x[int(2 * n / 3):]) - np.mean(cam_x[:third])) * 1000
    drift_y = float(np.mean(cam_y[int(2 * n / 3):]) - np.mean(cam_y[:third])) * 1000

    # 标准差
    std_x = float(np.std(cam_x)) * 1000
    std_y = float(np.std(cam_y)) * 1000

    return {
        'video': video_name,
        'frames': len(raw),
        'scale_um_per_px': scale,
        'radius_mean': radius_mean,
        'radius_cv_pct': radius_cv,
        'detection_success_pct': detection_rate,
        'amplitude_x_mm': amp_x,
        'amplitude_y_mm': amp_y,
        'amplitude_total_mm': amp_total,
        'amplitude_x_detrended_mm': amp_x_dt,
        'amplitude_y_detrended_mm': amp_y_dt,
        'amplitude_total_detrended_mm': amp_total_dt,
        'drift_x_mm': drift_x,
        'drift_y_mm': drift_y,
        'std_x_mm': std_x,
        'std_y_mm': std_y,
    }


def compute_improvement(before_val, after_val, lower_is_better=True):
    """计算改善百分比。"""
    if before_val is None or after_val is None or np.isnan(before_val) or np.isnan(after_val):
        return np.nan
    if abs(before_val) < 1e-10:
        return np.nan
    change = (after_val - before_val) / abs(before_val) * 100
    if lower_is_better:
        return -change  # 正值=改善（降低了），负值=恶化（升高了）
    return change  # 正值=改善（升高了），负值=恶化（降低了）


def build_comparison_table(before_root, after_root, video_names):
    """构建对比表。"""
    records = []
    for vname in video_names:
        before = load_video_metrics(before_root, vname)
        after = load_video_metrics(after_root, vname)
        if before is None and after is None:
            print(f"  [SKIP] {vname}: 修复前后数据均缺失")
            continue

        record = {'Video': vname}
        metrics = [
            'scale_um_per_px', 'radius_mean', 'radius_cv_pct',
            'detection_success_pct',
            'amplitude_x_mm', 'amplitude_y_mm', 'amplitude_total_mm',
            'amplitude_x_detrended_mm', 'amplitude_y_detrended_mm',
            'amplitude_total_detrended_mm',
            'drift_x_mm', 'drift_y_mm', 'std_x_mm', 'std_y_mm',
        ]
        for m in metrics:
            b_val = before[m] if before else np.nan
            a_val = after[m] if after else np.nan
            record[f'{m}_before'] = b_val
            record[f'{m}_after'] = a_val
            # 改善方向: scale越接近60越好(特殊), radius_cv越低越好, detection越高越好
            if m == 'detection_success_pct':
                record[f'{m}_improve'] = compute_improvement(b_val, a_val, lower_is_better=False)
            elif m == 'scale_um_per_px':
                # scale 改善 = 偏差缩小
                b_err = abs(b_val - THEORETICAL_SCALE) if not np.isnan(b_val) else np.nan
                a_err = abs(a_val - THEORETICAL_SCALE) if not np.isnan(a_val) else np.nan
                record[f'{m}_improve'] = compute_improvement(b_err, a_err, lower_is_better=True)
            else:
                record[f'{m}_improve'] = compute_improvement(b_val, a_val, lower_is_better=True)
        records.append(record)

    return pd.DataFrame(records)


def plot_comparison(df, output_dir):
    """生成对比图表（4张PNG）。"""
    videos = df['Video'].values
    x = np.arange(len(videos))
    width = 0.35

    # ---- 图1: scale 对比 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    b_vals = df['scale_um_per_px_before'].values
    a_vals = df['scale_um_per_px_after'].values
    ax.bar(x - width / 2, b_vals, width, label='修复前', color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, a_vals, width, label='修复后', color='coral', alpha=0.8)
    ax.axhline(THEORETICAL_SCALE, color='green', linestyle='--', linewidth=2, label=f'理论值 {THEORETICAL_SCALE}')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=45)
    ax.set_ylabel('Scale (μm/px)')
    ax.set_title('比例尺对比 (Scale, μm/px)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_scale.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图2: 振幅对比（原始 vs 去漂移）----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, direction, color in [(axes[0], 'x', 'steelblue'), (axes[1], 'y', 'coral')]:
        b_raw = df[f'amplitude_{direction}_mm_before'].values
        a_raw = df[f'amplitude_{direction}_mm_after'].values
        b_dt = df[f'amplitude_{direction}_detrended_mm_before'].values
        a_dt = df[f'amplitude_{direction}_detrended_mm_after'].values
        ax.bar(x - width * 1.5, b_raw, width, label='修复前(原始)', color=color, alpha=0.4)
        ax.bar(x - width / 2, b_dt, width, label='修复前(去漂移)', color=color, alpha=0.8)
        ax.bar(x + width / 2, a_raw, width, label='修复后(原始)', color='green', alpha=0.4)
        ax.bar(x + width * 1.5, a_dt, width, label='修复后(去漂移)', color='green', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(videos, rotation=45)
        ax.set_ylabel('Amplitude (mm)')
        ax.set_title(f'{direction.upper()} 方向振幅对比')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_amplitude.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图3: radius_cv% 和检测成功率 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # CV%
    ax = axes[0]
    b_cv = df['radius_cv_pct_before'].values
    a_cv = df['radius_cv_pct_after'].values
    ax.bar(x - width / 2, b_cv, width, label='修复前', color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, a_cv, width, label='修复后', color='coral', alpha=0.8)
    ax.axhline(5, color='red', linestyle='--', linewidth=1, label='目标 < 5%')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=45)
    ax.set_ylabel('CV (%)')
    ax.set_title('半径变异系数 (CV%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 检测成功率
    ax = axes[1]
    b_det = df['detection_success_pct_before'].values
    a_det = df['detection_success_pct_after'].values
    ax.bar(x - width / 2, b_det, width, label='修复前', color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, a_det, width, label='修复后', color='coral', alpha=0.8)
    ax.axhline(100, color='red', linestyle='--', linewidth=1, label='目标 100%')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=45)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('检测成功率 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_stability.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图4: 漂移量对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, direction, color in [(axes[0], 'x', 'steelblue'), (axes[1], 'y', 'coral')]:
        b_drift = df[f'drift_{direction}_mm_before'].values
        a_drift = df[f'drift_{direction}_mm_after'].values
        ax.bar(x - width / 2, b_drift, width, label='修复前', color=color, alpha=0.8)
        ax.bar(x + width / 2, a_drift, width, label='修复后', color='green', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(videos, rotation=45)
        ax.set_ylabel('Drift (mm)')
        ax.set_title(f'{direction.upper()} 方向漂移量')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_drift.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [OK] 4张对比图表已生成")


def generate_markdown_report(df, output_dir):
    """生成 Markdown 对比报告。"""
    lines = []
    lines.append("# 修复前后视频测试对比报告\n")
    lines.append(f"**测试视频**: {', '.join(df['Video'].values)}")
    lines.append(f"**排除视频**: 90, 100")
    lines.append(f"**理论 scale**: {THEORETICAL_SCALE} μm/px\n")

    # ---- 总览表 ----
    lines.append("## 一、总览对比表\n")
    lines.append("| 视频 | Scale前 | Scale后 | CV%前 | CV%后 | 检测率前 | 检测率后 | 总振幅前(mm) | 总振幅后(mm) | 去漂移后(mm) |")
    lines.append("|------|---------|---------|-------|-------|---------|---------|-------------|-------------|-------------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Video']} "
            f"| {row['scale_um_per_px_before']:.1f} | {row['scale_um_per_px_after']:.1f} "
            f"| {row['radius_cv_pct_before']:.1f} | {row['radius_cv_pct_after']:.1f} "
            f"| {row['detection_success_pct_before']:.0f} | {row['detection_success_pct_after']:.0f} "
            f"| {row['amplitude_total_mm_before']:.4f} | {row['amplitude_total_mm_after']:.4f} "
            f"| {row['amplitude_total_detrended_mm_after']:.4f} |"
        )
    lines.append("")

    # ---- 关键指标改善 ----
    lines.append("## 二、关键指标改善\n")

    # Scale
    lines.append("### 2.1 比例尺稳定性 (Scale, μm/px)\n")
    lines.append(f"- 理论值: {THEORETICAL_SCALE} μm/px")
    b_scales = df['scale_um_per_px_before'].dropna()
    a_scales = df['scale_um_per_px_after'].dropna()
    lines.append(f"- 修复前范围: {b_scales.min():.1f} ~ {b_scales.max():.1f} (标准差 {b_scales.std():.2f})")
    lines.append(f"- 修复后范围: {a_scales.min():.1f} ~ {a_scales.max():.1f} (标准差 {a_scales.std():.2f})")
    b_errs = (b_scales - THEORETICAL_SCALE).abs() / THEORETICAL_SCALE * 100
    a_errs = (a_scales - THEORETICAL_SCALE).abs() / THEORETICAL_SCALE * 100
    lines.append(f"- 修复前偏差: {b_errs.min():.1f}% ~ {b_errs.max():.1f}%")
    lines.append(f"- 修复后偏差: {a_errs.min():.1f}% ~ {a_errs.max():.1f}%\n")

    # Radius CV
    lines.append("### 2.2 半径稳定性 (CV%)\n")
    b_cvs = df['radius_cv_pct_before'].dropna()
    a_cvs = df['radius_cv_pct_after'].dropna()
    lines.append(f"- 修复前范围: {b_cvs.min():.1f}% ~ {b_cvs.max():.1f}%")
    lines.append(f"- 修复后范围: {a_cvs.min():.1f}% ~ {a_cvs.max():.1f}%")
    lines.append(f"- 目标: 全部 < 5%\n")

    # 检测成功率
    lines.append("### 2.3 检测成功率\n")
    b_dets = df['detection_success_pct_before'].dropna()
    a_dets = df['detection_success_pct_after'].dropna()
    lines.append(f"- 修复前范围: {b_dets.min():.0f}% ~ {b_dets.max():.0f}%")
    lines.append(f"- 修复后范围: {a_dets.min():.0f}% ~ {a_dets.max():.0f}%")
    lines.append(f"- 目标: 全部 = 100%\n")

    # 振幅
    lines.append("### 2.4 振幅修正\n")
    lines.append("| 视频 | 原始X前 | 原始X后 | 去漂移X前 | 去漂移X后 | 原始Y前 | 原始Y后 | 去漂移Y前 | 去漂移Y后 |")
    lines.append("|------|---------|---------|----------|----------|---------|---------|----------|----------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Video']} "
            f"| {row['amplitude_x_mm_before']:.4f} | {row['amplitude_x_mm_after']:.4f} "
            f"| {row['amplitude_x_detrended_mm_before']:.4f} | {row['amplitude_x_detrended_mm_after']:.4f} "
            f"| {row['amplitude_y_mm_before']:.4f} | {row['amplitude_y_mm_after']:.4f} "
            f"| {row['amplitude_y_detrended_mm_before']:.4f} | {row['amplitude_y_detrended_mm_after']:.4f} |"
        )
    lines.append("")

    # 漂移
    lines.append("### 2.5 漂移量\n")
    lines.append("| 视频 | X漂移前 | X漂移后 | Y漂移前 | Y漂移后 |")
    lines.append("|------|---------|---------|---------|---------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Video']} "
            f"| {row['drift_x_mm_before']:.4f} | {row['drift_x_mm_after']:.4f} "
            f"| {row['drift_y_mm_before']:.4f} | {row['drift_y_mm_after']:.4f} |"
        )
    lines.append("")

    # ---- 结论 ----
    lines.append("## 三、结论\n")
    # 自动判断通过/失败
    scale_pass = all(abs(v - THEORETICAL_SCALE) / THEORETICAL_SCALE < 0.05
                     for v in a_scales if not np.isnan(v))
    cv_pass = all(v < 5 for v in a_cvs if not np.isnan(v))
    det_pass = all(v >= 99.9 for v in a_dets if not np.isnan(v))

    lines.append(f"- [{'x' if scale_pass else ' '}] Scale 偏差全部 < 5%: {'通过' if scale_pass else '未通过'}")
    lines.append(f"- [{'x' if cv_pass else ' '}] Radius CV% 全部 < 5%: {'通过' if cv_pass else '未通过'}")
    lines.append(f"- [{'x' if det_pass else ' '}] 检测成功率全部 = 100%: {'通过' if det_pass else '未通过'}")

    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] Markdown 报告已生成: {report_path}")


def save_excel(df, output_dir):
    """保存完整指标表到 Excel。"""
    xlsx_path = os.path.join(output_dir, 'comparison_metrics.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comparison', index=False)

        # 聚合统计
        metric_cols = [c for c in df.columns if c not in ['Video']]
        summary_data = []
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary_data.append({
                    'Metric': col,
                    'Mean': vals.mean(),
                    'Std': vals.std(),
                    'Min': vals.min(),
                    'Max': vals.max(),
                })
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

    print(f"  [OK] Excel 指标表已生成: {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="修复前后视频测试对比")
    parser.add_argument("--before", type=str, default="test/output",
                        help="修复前输出目录（默认 test/output）")
    parser.add_argument("--after", type=str, default="test/output_fixed",
                        help="修复后输出目录（默认 test/output_fixed）")
    parser.add_argument("--exclude", type=str, default="90,100",
                        help="排除的视频（逗号分隔，默认 90,100）")
    parser.add_argument("--output", type=str, default=None,
                        help="报告输出目录（默认 {after}/analysis）")
    args = parser.parse_args()

    before_root = os.path.join(PROJECT_ROOT, args.before)
    after_root = os.path.join(PROJECT_ROOT, args.after)
    output_dir = args.output if args.output else os.path.join(after_root, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"修复前数据: {before_root}")
    print(f"修复后数据: {after_root}")
    print(f"排除视频: {args.exclude}")
    print(f"报告输出: {output_dir}\n")

    # 找到修复后处理的视频
    video_names = find_video_dirs(after_root, args.exclude)
    if not video_names:
        print("[ERROR] 修复后目录中未找到视频数据，请先运行 batch_analysis")
        sys.exit(1)

    print(f"找到 {len(video_names)} 个视频: {video_names}\n")

    # 构建对比表
    print("计算指标中...")
    df = build_comparison_table(before_root, after_root, video_names)

    if df.empty:
        print("[ERROR] 无有效对比数据")
        sys.exit(1)

    # 生成输出
    print("\n生成报告中...")
    save_excel(df, output_dir)
    plot_comparison(df, output_dir)
    generate_markdown_report(df, output_dir)

    print(f"\n{'='*50}")
    print(f"对比报告生成完成!")
    print(f"{'='*50}")
    print(f"  Markdown: {os.path.join(output_dir, 'comparison_report.md')}")
    print(f"  Excel:    {os.path.join(output_dir, 'comparison_metrics.xlsx')}")
    print(f"  图表:     {output_dir}/comparison_*.png")


if __name__ == "__main__":
    main()
