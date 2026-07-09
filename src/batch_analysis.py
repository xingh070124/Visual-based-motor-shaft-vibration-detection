"""批量电机轴振动分析：处理 test/video/ 下所有视频。

对每个视频：
    1. 提取首帧 → 用户选择 ROI（首次选择后可复用）
    2. STARK-ST 跟踪 + 轮廓圆心检测（或椭圆检测+倾斜补偿）
    3. 像素坐标 → 相机坐标转换（等向性或各向异性 scale）
    4. 输出 Excel 数据
    5. 绘制振动曲线图 (PNG)
    6. 渲染带跟踪标记的输出视频 (MP4)

用法：
    python -m src.batch_analysis                # 交互式选择 ROI
    python -m src.batch_analysis --no-stark     # 使用 OpenCV CSRT 备选跟踪器
    python -m src.batch_analysis --skip-render  # 跳过视频渲染（仅输出数据和图）
    python -m src.batch_analysis --ellipse-correct  # 启用椭圆检测+倾斜补偿
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import deque
import glob
import argparse

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.tracking.shaft_detector import (
    detect_shaft_center_in_roi,
    estimate_expected_radius_from_roi,
    detect_ellipse_in_roi
)
from src.tracking.coordinate import (
    compute_scale_m_per_px,
    pixel_to_mm,
    estimate_tilt_angle,
    pixel_to_mm_anisotropic
)
from src.vibration.analyzer import (
    compute_vibration_amplitude,
    print_vibration_result
)


# =============================================================================
# 中文字体配置
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 辅助函数
# =============================================================================

def natural_key(s: str) -> list:
    """自然排序键函数。"""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def get_video_files(folder: str) -> List[str]:
    """获取指定文件夹下所有视频文件，自然排序。"""
    patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV", "*.MP4"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files), key=natural_key)


def auto_detect_shaft_roi(
    frame: np.ndarray,
    expected_radius: Optional[float] = None,
    margin_ratio: float = 1.3
) -> Optional[List[int]]:
    """自动检测首帧中电机轴的 ROI。

    使用霍夫圆检测定位轴心，返回以之为中心的方形 ROI。

    Args:
        frame: 首帧图像 (BGR)
        expected_radius: 预期像素半径（可选），用于评分 + ROI 尺寸估算。
                         None 时仅按中心距离评分，ROI 尺寸用检测半径。
        margin_ratio: ROI 边长为半径的倍数，默认 1.3

    Returns:
        [x, y, w, h] ROI，未找到则返回 None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = frame.shape[:2]

    # 霍夫圆检测 — 半径范围基于图像尺寸（不再依赖 expected_radius→Z）
    max_img_radius = min(w, h) // 3
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=30,
        minRadius=5,
        maxRadius=max_img_radius
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    best = None
    best_score = float('inf')

    # 选择靠近图像中心（且接近预期半径，如有）的圆
    img_center = np.array([w / 2, h / 2])

    for c in circles[0, :]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        # 评分 = 半径偏差 + 距离中心偏差（归一化）
        center_dist = np.linalg.norm(np.array([cx, cy]) - img_center) / max(w, h)
        radius_err = 0.0
        if expected_radius is not None:
            radius_err = abs(r - expected_radius) / expected_radius
        score = radius_err + center_dist * 2

        if score < best_score:
            best_score = score
            best = (cx, cy, r)

    if best is None or best_score > 1.5:
        return None

    cx, cy, r = best
    # ROI 尺寸：优先用 expected_radius，否则用检测半径
    roi_ref = expected_radius if expected_radius is not None else float(r)
    half_size = int(roi_ref * margin_ratio)
    x = max(0, cx - half_size)
    y = max(0, cy - half_size)
    size = min(half_size * 2, min(w - x, h - y))

    roi = [x, y, size, size]
    print(f"[AUTO] 检测到电机轴: 中心=({cx}, {cy}), 检测半径={r} px, "
          f"预期={expected_radius if expected_radius is not None else 'N/A'}, ROI={roi}")
    return roi


# =============================================================================
# STARK 跟踪器封装（带备选）
# =============================================================================

class OpenCVTracker:
    """OpenCV CSRT 备选跟踪器，当 STARK 不可用时使用。"""

    def __init__(self):
        self._tracker = None
        self._initialized = False

    def initialize(self, first_frame, init_bbox: list):
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(first_frame, tuple(init_bbox))
        self._initialized = True
        return {}

    def track(self, frame):
        success, bbox = self._tracker.update(frame)
        if success:
            x, y, w, h = [float(v) for v in bbox]
            return {'target_bbox': [x, y, w, h]}
        return {'target_bbox': [0, 0, 100, 100]}

    @property
    def is_initialized(self):
        return self._initialized


def create_tracker(use_stark: bool = True):
    """创建跟踪器（STARK 或 OpenCV CSRT 备选）。"""
    if use_stark:
        try:
            from src.tracking.stark_wrapper import StarkTracker
            tracker = StarkTracker(
                checkpoint_rel_path=config.CHECKPOINT_REL_PATH,
                params_name=config.CHECKPOINT_PARAM_NAME,
                dataset_name=config.CHECKPOINT_DATASET_NAME
            )
            print("[OK] 使用 STARK-ST 跟踪器")
            return tracker
        except Exception as e:
            print(f"[WARN] STARK 加载失败: {e}")
            print("[WARN] 回退到 OpenCV CSRT 跟踪器")
    else:
        print("[OK] 使用 OpenCV CSRT 跟踪器")
    return OpenCVTracker()


# =============================================================================
# 绘图函数
# =============================================================================

def plot_vibration_curves(
    df: pd.DataFrame,
    video_name: str,
    output_dir: str,
    vibration: Dict
):
    """绘制振动分析曲线图。

    Args:
        df: 含 Frame, Cam_X, Cam_Y 的 DataFrame
        video_name: 视频名称（用于标题）
        output_dir: 输出目录
        vibration: 振动分析结果字典
    """
    frames = df["Frame"].values
    cam_x = df["Cam_X (m)"].values * 1000  # 转换为 mm
    cam_y = df["Cam_Y (m)"].values * 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Motor Shaft Vibration Analysis — {video_name}", fontsize=14)

    # ---- 子图1: X 方向位移 ----
    ax1 = axes[0, 0]
    ax1.plot(frames, cam_x, 'b-', linewidth=0.8, alpha=0.8)
    ax1.axhline(np.mean(cam_x), color='r', linestyle='--', linewidth=1, label=f'Mean={np.mean(cam_x):.4f} mm')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Displacement (mm)')
    ax1.set_title(f'X-Direction Displacement (Amp={vibration["amplitude_x_mm"]:.4f} mm)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 子图2: Y 方向位移 ----
    ax2 = axes[0, 1]
    ax2.plot(frames, cam_y, 'r-', linewidth=0.8, alpha=0.8)
    ax2.axhline(np.mean(cam_y), color='b', linestyle='--', linewidth=1, label=f'Mean={np.mean(cam_y):.4f} mm')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Displacement (mm)')
    ax2.set_title(f'Y-Direction Displacement (Amp={vibration["amplitude_y_mm"]:.4f} mm)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- 子图3: 轨迹散点图 ----
    ax3 = axes[1, 0]
    scatter = ax3.scatter(cam_x, cam_y, c=frames, cmap='viridis', s=2, alpha=0.6)
    ax3.set_xlabel('X Displacement (mm)')
    ax3.set_ylabel('Y Displacement (mm)')
    ax3.set_title(f'Trajectory (Total Amp={vibration["amplitude_total_mm"]:.4f} mm)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Frame')

    # ---- 子图4: 统计信息 ----
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = (
        f"Statistics Summary\n"
        f"{'='*30}\n"
        f"Samples:         {vibration['n_samples']}\n"
        f"X Amplitude:     {vibration['amplitude_x_mm']:.4f} mm\n"
        f"Y Amplitude:     {vibration['amplitude_y_mm']:.4f} mm\n"
        f"Total Amplitude: {vibration['amplitude_total_mm']:.4f} mm\n"
        f"{'-'*30}\n"
        f"X Mean:          {vibration['mean_x']*1000:.4f} mm\n"
        f"Y Mean:          {vibration['mean_y']*1000:.4f} mm\n"
        f"X Std:           {vibration['std_x']*1000:.4f} mm\n"
        f"Y Std:           {vibration['std_y']*1000:.4f} mm\n"
    )
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='center')

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{video_name}_vibration.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 振动曲线图已保存: {png_path}")
    return png_path


# =============================================================================
# 单视频处理
# =============================================================================

def process_single_video(
    video_path: str,
    output_dir: str,
    init_roi: Optional[List[int]],
    tracker_factory,
    skip_render: bool = False,
    step: int = 1,
    use_ellipse_correction: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[List[int]]]:
    """处理单个视频，返回 (DataFrame, vibration_dict, used_roi)。

    比例尺通过轴径自标定：用 ROI 估算检测先验 → 前 N 帧中位数半径 → scale。
    当 use_ellipse_correction=True 时，使用椭圆检测 + 各向异性 scale 补偿倾斜误差。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        init_roi: 初始 ROI [x, y, w, h]，None 则交互式选择
        tracker_factory: 无参函数，返回跟踪器实例
        skip_render: 是否跳过视频渲染
        step: 帧采样步长
        use_ellipse_correction: 是否启用椭圆检测+倾斜补偿

    Returns:
        (df, vibration, used_roi)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"处理: {video_name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率: {frame_w}x{frame_h}, FPS: {fps:.2f}, 总帧数: {total_frames}")

    # ---- 读取首帧 ----
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] 无法读取首帧")
        cap.release()
        return None, None, None

    # ---- ROI 选择 ----
    if init_roi is not None:
        used_roi = init_roi
        print(f"[OK] 使用预设 ROI: {used_roi}")
    else:
        print("\n>>> 请在图像上框选电机轴区域，按 ENTER 确认，按 C 取消 <<<")
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        # 缩小窗口以适配屏幕
        cv2.resizeWindow("Select ROI", frame_w // 2, frame_h // 2)
        roi = cv2.selectROI("Select ROI", first_frame, False, False)
        cv2.destroyWindow("Select ROI")
        if sum(roi) == 0:
            print("[WARN] ROI 选择已取消，跳过此视频")
            cap.release()
            return None, None, None
        used_roi = [int(v) for v in roi]
        print(f"[OK] 已选择 ROI: {used_roi}")

    # ---- 首帧圆心检测（ROI 先验 → 自适应） ----
    expected_radius = estimate_expected_radius_from_roi(used_roi)

    # 椭圆模式参数初始化
    init_a = 0.0
    init_b = 0.0
    init_angle = 0.0
    init_theta = 0.0

    if use_ellipse_correction:
        init_center_x, init_center_y, init_a, init_b, init_angle = \
            detect_ellipse_in_roi(
                first_frame, used_roi,
                expected_radius_pixels=expected_radius,
                circularity_threshold=config.CIRCULARITY_THRESHOLD,
                min_contour_area=config.MIN_CONTOUR_AREA,
                canny_low=config.CANNY_LOW,
                canny_high=config.CANNY_HIGH,
                blur_kernel=config.GAUSSIAN_BLUR_KERNEL,
                return_params=True
            )
        init_theta = estimate_tilt_angle(init_a, init_b)
        # 用半长轴作为半径先验
        init_radius = init_a if init_a > 0 else expected_radius
        if init_a > 0:
            expected_radius = init_a
        print(f"  [ELLIPSE] 首帧: a={init_a:.1f}, b={init_b:.1f}, "
              f"angle={init_angle:.1f}°, θ_est={init_theta:.2f}°")
    else:
        init_center_x, init_center_y, init_radius = detect_shaft_center_in_roi(
            first_frame, used_roi,
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
            expected_radius = init_radius  # 首帧检测半径作为先验

    # 滑动窗口中位数（替代逐帧自适应更新，防止正反馈崩溃）
    radius_history = deque(maxlen=config.RADIUS_HISTORY_WINDOW)
    init_radius_for_scale = init_radius if init_radius > 0 else expected_radius  # 比例尺标定用
    init_center = (init_center_x, init_center_y)  # 返回 2-tuple 给后续 old-style ref

    # ---- 初始化跟踪器 ----
    tracker = tracker_factory()
    try:
        tracker.initialize(first_frame, used_roi)
    except Exception as e:
        print(f"[ERROR] 跟踪器初始化失败: {e}")
        cap.release()
        return None, None, used_roi

    # ---- 初始化 VideoWriter ----
    writer = None
    if not skip_render:
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        out_fps = fps if config.OUTPUT_VIDEO_FPS is None else config.OUTPUT_VIDEO_FPS
        # MPEG4 不支持 >65535 的 timebase denominator，高 FPS 需降至 60
        if out_fps > 60:
            print(f"  [INFO] FPS {out_fps:.2f} 过高，输出降至 30 FPS 以确保兼容")
            out_fps = 30.0

        # 尝试多种编码器
        codecs = [
            ('avc1', '.mp4'),   # H.264
            ('mp4v', '.mp4'),   # MPEG-4
            ('XVID', '.avi'),   # Xvid AVI
            ('MJPG', '.avi'),   # Motion JPEG AVI
        ]
        writer = None
        for codec, ext in codecs:
            try:
                ext_path = output_video_path.replace('.mp4', ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                w = cv2.VideoWriter(ext_path, fourcc, out_fps, (frame_w, frame_h))
                if w.isOpened():
                    writer = w
                    output_video_path = ext_path
                    print(f"  [OK] 输出视频 ({codec}): {output_video_path}")
                    break
                else:
                    w.release()
            except Exception:
                continue

        if writer is None:
            print(f"  [WARN] 无法创建视频编码器，跳过视频输出")
            skip_render = True

    # ---- 逐帧处理 ----
    results = []  # 每项: [frame_idx, pixel_x, pixel_y, radius]（原始，延迟换算）
    frame_idx = 1

    # 跟踪状态（用于非采样帧的渲染回填）
    last_bbox = used_roi
    last_cx = used_roi[0] + used_roi[2] / 2
    last_cy = used_roi[1] + used_roi[3] / 2
    last_radius = expected_radius
    # 椭圆模式渲染状态
    last_a = 0.0
    last_b = 0.0
    last_angle = 0.0
    last_theta = 0.0

    # 首帧：用 STARK bbox 中心作为初始位置
    init_px = last_cx
    init_py = last_cy

    if use_ellipse_correction:
        # 椭圆模式：首帧检测已在上面完成，直接使用 init_a/init_b/init_angle
        vis_a = init_a
        vis_b = init_b
        vis_angle = init_angle
        vis_theta = init_theta
        last_radius = init_a if init_a > 0 else last_radius
        results.append([1, init_px, init_py, vis_a, vis_b, vis_angle])
    else:
        # 圆形模式：首帧额外检测用于显示
        _, _, vis_r = detect_shaft_center_in_roi(
            first_frame, used_roi, expected_radius_pixels=expected_radius,
            circularity_threshold=0.85,
            radius_tolerance=0.25,
            min_contour_area=config.MIN_CONTOUR_AREA,
            canny_low=config.CANNY_LOW, canny_high=config.CANNY_HIGH,
            blur_kernel=config.GAUSSIAN_BLUR_KERNEL, return_radius=True
        )
        last_radius = vis_r if vis_r > 0 else last_radius
        results.append([1, init_px, init_py, vis_r])

    # 渲染首帧
    if writer is not None:
        if use_ellipse_correction:
            ellipse_info = {'a': vis_a, 'b': vis_b, 'angle': vis_angle,
                            'theta_est': vis_theta}
        else:
            ellipse_info = None
        rendered = _render_frame(
            first_frame, used_roi, last_cx, last_cy, last_radius, 1,
            ellipse_info=ellipse_info
        )
        writer.write(rendered)

    print(f"  处理中... (步长={step}, 0/{total_frames})", end='', flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 帧采样：跳过不需要处理的帧
        if (frame_idx - 1) % step != 0:
            if writer is not None:
                if use_ellipse_correction:
                    skip_ellipse = {'a': last_a, 'b': last_b,
                                    'angle': last_angle, 'theta_est': last_theta}
                else:
                    skip_ellipse = None
                rendered = _render_frame(
                    frame, last_bbox, last_cx, last_cy, last_radius, frame_idx,
                    ellipse_info=skip_ellipse
                )
                writer.write(rendered)
            continue

        # STARK 跟踪 → bbox
        try:
            state = tracker.track(frame)
            pred_box = state['target_bbox']
        except Exception:
            pred_box = last_bbox

        # ★ 振动数据：直接用 STARK bbox 中心（更可靠）
        track_cx = pred_box[0] + pred_box[2] / 2
        track_cy = pred_box[1] + pred_box[3] / 2

        if use_ellipse_correction:
            # 椭圆检测：获取长短轴和倾角
            _, _, vis_a, vis_b, vis_angle = detect_ellipse_in_roi(
                frame, pred_box,
                expected_radius_pixels=expected_radius,
                circularity_threshold=0.3,
                min_contour_area=config.MIN_CONTOUR_AREA,
                canny_low=config.CANNY_LOW, canny_high=config.CANNY_HIGH,
                blur_kernel=config.GAUSSIAN_BLUR_KERNEL,
                return_params=True
            )
            vis_theta = estimate_tilt_angle(vis_a, vis_b)

            # 半长轴用于半径历史
            vis_radius = vis_a

            # 更新椭圆渲染状态
            if vis_a > 0:
                last_a = vis_a
                last_b = vis_b
                last_angle = vis_angle
                last_theta = vis_theta
        else:
            # 圆检测：用于可视化渲染 + 半径基准
            _, _, vis_radius = detect_shaft_center_in_roi(
                frame, pred_box, expected_radius_pixels=expected_radius,
                circularity_threshold=0.85,
                radius_tolerance=0.25,
                min_contour_area=config.MIN_CONTOUR_AREA,
                canny_low=config.CANNY_LOW, canny_high=config.CANNY_HIGH,
                blur_kernel=config.GAUSSIAN_BLUR_KERNEL, return_radius=True
            )

        # 滑动窗口中位数更新（替代逐帧自适应，防止正反馈崩溃）
        if vis_radius > 0:
            # 异常检测：偏离首帧半径超过阈值则跳过
            if abs(vis_radius - init_radius_for_scale) / max(init_radius_for_scale, 1) < config.RADIUS_ANOMALY_THRESHOLD:
                radius_history.append(vis_radius)
            # 用中位数更新expected_radius（至少需要10帧）
            if len(radius_history) >= config.RADIUS_HISTORY_MIN:
                expected_radius = float(np.median(list(radius_history)))

        # 更新渲染状态
        last_bbox = pred_box
        last_cx, last_cy = track_cx, track_cy
        last_radius = vis_radius if vis_radius > 0 else last_radius

        # 存原始数据（延迟到标定 scale 后再换算）
        if use_ellipse_correction:
            results.append([frame_idx, track_cx, track_cy,
                           vis_a, vis_b, vis_angle])
        else:
            results.append([frame_idx, track_cx, track_cy, vis_radius])

        # 渲染
        if writer is not None:
            if use_ellipse_correction:
                frame_ellipse = {'a': last_a, 'b': last_b,
                                 'angle': last_angle, 'theta_est': last_theta}
            else:
                frame_ellipse = None
            rendered = _render_frame(
                frame, pred_box, track_cx, track_cy, last_radius, frame_idx,
                ellipse_info=frame_ellipse
            )
            writer.write(rendered)

        # 进度
        if len(results) % 50 == 0:
            print(f"\r  处理中... ({frame_idx}/{total_frames}, 采样{len(results)}帧)", end='', flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\r  处理完成! ({frame_idx}/{total_frames} 帧)")

    # ---- 首帧比例尺标定 + 批量坐标换算 ----
    scale = None
    scale_major = None
    scale_minor = None

    if use_ellipse_correction:
        # 各向异性 scale：长轴 D/(2a)，短轴 D/(2b)
        if init_a > 0 and init_b > 0:
            scale_major = config.SHAFT_DIAMETER_M / (2.0 * init_a)
            scale_minor = config.SHAFT_DIAMETER_M / (2.0 * init_b)
            scale = scale_major  # 用于 Summary 输出
            print(
                f"  [CALIB] 各向异性比例尺: a={init_a:.1f}px, b={init_b:.1f}px → "
                f"scale_major={scale_major*1e6:.1f} μm/px, "
                f"scale_minor={scale_minor*1e6:.1f} μm/px "
                f"(θ_est={init_theta:.2f}°)"
            )
            for row in results:
                # row: [frame_idx, px, py, a, b, angle]
                px, py, ell_angle = row[1], row[2], row[5]
                cam_x, cam_y = pixel_to_mm_anisotropic(
                    px, py, config.CX, config.CY,
                    scale_major, scale_minor, ell_angle
                )
                row.append(cam_x)
                row.append(cam_y)
        else:
            print("  [WARN] 首帧椭圆检测失败，换算失败")
            for row in results:
                row.append(0.0)
                row.append(0.0)
    else:
        # 等向性 scale：D/(2R)
        if init_radius_for_scale > 0:
            scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, init_radius_for_scale)
            print(
                f"  [CALIB] 比例尺: 首帧半径={init_radius_for_scale:.1f} px → "
                f"scale={scale*1e6:.1f} μm/px "
                f"(轴径{config.SHAFT_DIAMETER_MM:.0f}mm 为唯一先验，不依赖 Z)"
            )
            for row in results:
                cam_x, cam_y = pixel_to_mm(row[1], row[2], config.CX, config.CY, scale)
                row.append(cam_x)
                row.append(cam_y)
        else:
            print("  [WARN] 所有帧均未检测到圆，换算失败")
            for row in results:
                row.append(0.0)
                row.append(0.0)

    # ---- 保存 Excel ----
    if use_ellipse_correction:
        df = pd.DataFrame(
            results,
            columns=["Frame", "Pixel_X", "Pixel_Y", "SemiMajor (px)",
                     "SemiMinor (px)", "EllipseAngle (deg)",
                     "Cam_X (m)", "Cam_Y (m)"]
        )
    else:
        df = pd.DataFrame(
            results,
            columns=["Frame", "Pixel_X", "Pixel_Y", "Radius (px)",
                     "Cam_X (m)", "Cam_Y (m)"]
        )
    xlsx_path = os.path.join(output_dir, f"{video_name}_data.xlsx")

    # 振动分析
    vibration = compute_vibration_amplitude(
        df["Cam_X (m)"].values, df["Cam_Y (m)"].values
    )
    print_vibration_result(vibration)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer_xl:
        df.to_excel(writer_xl, sheet_name='Raw Data', index=False)

        # Summary sheet
        summary_rows = [
            ["Video", video_name],
            ["Total Frames", len(df)],
            ["FPS", f"{fps:.2f}"],
            ["Resolution", f"{frame_w}x{frame_h}"],
            ["ROI", str(used_roi)],
            ["Scale (μm/px)", f"{scale*1e6:.1f}" if scale else "N/A"],
        ]
        if use_ellipse_correction:
            summary_rows.extend([
                ["Mode", "Ellipse + Tilt Correction"],
                ["Semi-Major a (px)", f"{init_a:.2f}"],
                ["Semi-Minor b (px)", f"{init_b:.2f}"],
                ["Tilt θ_est (deg)", f"{init_theta:.2f}"],
                ["Scale Major (μm/px)", f"{scale_major*1e6:.1f}" if scale_major else "N/A"],
                ["Scale Minor (μm/px)", f"{scale_minor*1e6:.1f}" if scale_minor else "N/A"],
            ])
        summary_rows.extend([
            ["", ""],
            ["X Amplitude (mm)", f"{vibration['amplitude_x_mm']:.4f}"],
            ["Y Amplitude (mm)", f"{vibration['amplitude_y_mm']:.4f}"],
            ["Total Amplitude (mm)", f"{vibration['amplitude_total_mm']:.4f}"],
            ["X Mean (mm)", f"{vibration['mean_x']*1000:.4f}"],
            ["Y Mean (mm)", f"{vibration['mean_y']*1000:.4f}"],
            ["X Std (mm)", f"{vibration['std_x']*1000:.4f}"],
            ["Y Std (mm)", f"{vibration['std_y']*1000:.4f}"],
        ])
        summary = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])
        summary.to_excel(writer_xl, sheet_name='Summary', index=False)

    print(f"  [OK] 数据已保存: {xlsx_path}")

    # ---- 绘图 ----
    plot_vibration_curves(df, video_name, output_dir, vibration)

    return df, vibration, used_roi


def _render_frame(
    frame: np.ndarray,
    bbox,
    center_x: float,
    center_y: float,
    radius: float,
    frame_idx: int,
    ellipse_info: Optional[dict] = None
) -> np.ndarray:
    """在帧上绘制跟踪标记。

    标记内容：
        - 绿色矩形: STARK 跟踪 bbox
        - 红色实心圆: 检测圆心
        - 红色圆圈/蓝色椭圆: 检测轮廓
        - 黄色文字: 帧号、像素坐标、椭圆参数（倾斜模式）

    Args:
        ellipse_info: 椭圆参数字典，包含 a, b, angle, theta_est。
                      None 时绘制圆形，非 None 时绘制椭圆+倾斜信息。
    """
    rendered = frame.copy()
    cx, cy = int(center_x), int(center_y)
    r = int(radius)

    # 绿色 bbox
    x, y, w, h = map(int, bbox)
    cv2.rectangle(rendered, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 红色圆心
    cv2.circle(rendered, (cx, cy), 5, (0, 0, 255), -1)

    if ellipse_info is not None:
        # === 椭圆模式：绘制蓝色椭圆轮廓 ===
        a = ellipse_info.get('a', 0)
        b = ellipse_info.get('b', 0)
        angle = ellipse_info.get('angle', 0)
        theta = ellipse_info.get('theta_est', 0)

        if a > 0 and b > 0:
            # cv2.ellipse 需要全长轴（直径）
            cv2.ellipse(rendered, (cx, cy), (int(a * 2), int(b * 2)),
                        angle, 0, 360, (255, 128, 0), 2)

        # 倾斜方向指示线（长轴方向）
        rad = np.radians(angle)
        dx = int(a * np.cos(rad))
        dy = int(a * np.sin(rad))
        cv2.line(rendered, (cx - dx, cy - dy), (cx + dx, cy + dy),
                 (0, 200, 255), 1)

        # 信息文字
        info_lines = [
            f"Frame: {frame_idx}",
            f"Center: ({cx}, {cy}) px",
            f"Semi-axes: a={a:.1f}, b={b:.1f} px",
            f"Angle: {angle:.1f} deg",
            f"Tilt est: {theta:.2f} deg",
            f"[TILT CORRECTED]"
        ]
        y0 = 30
        for i, line in enumerate(info_lines):
            color = (0, 200, 255) if "TILT" in line else (0, 255, 255)
            cv2.putText(rendered, line, (10, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # === 圆形模式：绘制红色圆轮廓 ===
        if r > 0:
            cv2.circle(rendered, (cx, cy), r, (0, 0, 255), 2)

        info_lines = [
            f"Frame: {frame_idx}",
            f"Center: ({cx}, {cy}) px",
            f"Radius: {r} px"
        ]
        y0 = 30
        for i, line in enumerate(info_lines):
            cv2.putText(rendered, line, (10, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return rendered


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="批量电机轴振动分析")
    parser.add_argument("--no-stark", action="store_true",
                        help="使用 OpenCV CSRT 备选跟踪器（不使用 STARK）")
    parser.add_argument("--skip-render", action="store_true",
                        help="跳过视频渲染，仅输出数据和图表")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="预设 ROI [x y w h]（跳过自动/交互式选择）")
    parser.add_argument("--video", type=str, default=None,
                        help="仅处理指定视频文件（按文件名匹配）")
    parser.add_argument("--manual-roi", action="store_true",
                        help="手动选择 ROI（默认自动检测）")
    parser.add_argument("--step", type=int, default=1,
                        help="帧采样步长（默认1=每帧处理，设为3=每3帧处理一次，大幅提速）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有结果的视频")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="自定义输出目录（默认 test/output）")
    parser.add_argument("--exclude", type=str, default=None,
                        help="排除指定视频（逗号分隔视频名，如 90,100）")
    parser.add_argument("--ellipse-correct", action="store_true",
                        help="启用椭圆检测+倾斜补偿（各向异性 scale 修正轴倾斜误差）")
    args = parser.parse_args()

    # ---- 查找视频 ----
    video_folder = os.path.join(_PROJECT_ROOT, config.VIDEO_FOLDER)
    if not os.path.isdir(video_folder):
        video_folder = config.VIDEO_FOLDER

    video_files = get_video_files(video_folder)
    if args.video:
        # 精确匹配或按 RPM 数字匹配
        import re
        target = args.video
        video_files = [v for v in video_files
                       if os.path.splitext(os.path.basename(v))[0] == target]

    # 排除指定视频
    if args.exclude:
        excl_names = {x.strip() for x in args.exclude.split(',') if x.strip()}
        before = len(video_files)
        video_files = [v for v in video_files
                       if os.path.splitext(os.path.basename(v))[0] not in excl_names]
        print(f"[INFO] 排除视频 {sorted(excl_names)}，{before} -> {len(video_files)} 个视频")

    if not video_files:
        print(f"[ERROR] 在 {video_folder} 中未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")

    # ---- 输出目录 ----
    output_root = os.path.join(_PROJECT_ROOT, args.output_dir if args.output_dir else config.BATCH_OUTPUT_DIR)
    os.makedirs(output_root, exist_ok=True)
    print(f"\n输出目录: {output_root}")

    # ---- 相机参数（仅主点 cx/cy 参与换算，Z 和 fx 不再需要） ----
    print(f"内参主点: cx={config.CX:.2f}, cy={config.CY:.2f}")
    if args.ellipse_correct:
        print(f"[MODE] 椭圆检测+倾斜补偿模式: 启用")
        print(f"  - 检测: detect_ellipse_in_roi (fitEllipse)")
        print(f"  - 换算: pixel_to_mm_anisotropic (各向异性 scale)")
        print(f"  - 渲染: 椭圆轮廓 + 倾角θ + [TILT CORRECTED] 标记")
    else:
        print(f"比例尺标定: 轴径={config.SHAFT_DIAMETER_MM:.0f}mm, 首帧检测半径")

    # ---- ROI ----
    init_roi = args.roi if args.roi else None

    # 自动检测 ROI（如果未预设且非手动模式）
    if init_roi is None and not args.manual_roi:
        # 用第一个视频的首帧自动检测（不传 expected_radius，按图像中心距离评分）
        first_video = video_files[0]
        cap = cv2.VideoCapture(first_video)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            init_roi = auto_detect_shaft_roi(first_frame, expected_radius=None)
            if init_roi is not None:
                print(f"[OK] 自动检测 ROI: {init_roi}")
            else:
                print("[WARN] 自动检测失败，将使用手动选择")
        else:
            print("[WARN] 无法读取首帧，将使用手动选择")

    # 如果仍无 ROI 且非手动模式，回退到手动选择
    if init_roi is None and not args.manual_roi:
        print("[INFO] 回退到交互式 ROI 选择...")
        first_video = video_files[0]
        cap = cv2.VideoCapture(first_video)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            h, w = first_frame.shape[:2]
            cv2.resizeWindow("Select ROI", w // 2, h // 2)
            roi = cv2.selectROI("Select ROI", first_frame, False, False)
            cv2.destroyWindow("Select ROI")
            if sum(roi) > 0:
                init_roi = [int(v) for v in roi]
                print(f"[OK] 已选择 ROI: {init_roi}")

    # ---- 跟踪器工厂 ----
    use_stark = not args.no_stark

    def tracker_factory():
        return create_tracker(use_stark=use_stark)

    # ---- 逐视频处理 ----
    all_summaries = []
    shared_roi = init_roi

    for i, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_root, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # 断点续传：跳过已有完整输出的视频
        if args.resume:
            required = [f"{video_name}_data.xlsx", f"{video_name}_vibration.png"]
            if all(os.path.exists(os.path.join(video_output_dir, r)) for r in required):
                print(f"\n[SKIP] {video_name} 已有完整输出，跳过")
                # 读取已有数据补充汇总
                try:
                    existing_df = pd.read_excel(os.path.join(video_output_dir, f"{video_name}_data.xlsx"))
                    from src.vibration.analyzer import compute_vibration_amplitude
                    vib = compute_vibration_amplitude(
                        existing_df["Cam_X (m)"].values, existing_df["Cam_Y (m)"].values)
                    all_summaries.append({
                        "Video": video_name,
                        "Frames": len(existing_df),
                        "Mode": "Ellipse+Tilt" if "SemiMajor (px)" in existing_df.columns else "Circle",
                        "X_Amp_mm": vib['amplitude_x_mm'],
                        "Y_Amp_mm": vib['amplitude_y_mm'],
                        "Total_Amp_mm": vib['amplitude_total_mm'],
                        "X_Mean_mm": vib['mean_x'] * 1000,
                        "Y_Mean_mm": vib['mean_y'] * 1000,
                        "X_Std_mm": vib['std_x'] * 1000,
                        "Y_Std_mm": vib['std_y'] * 1000,
                    })
                    if shared_roi is None:
                        shared_roi = [709, 437, 434, 434]  # default
                except Exception:
                    pass
                continue

        df, vibration, used_roi = process_single_video(
            video_path=video_path,
            output_dir=video_output_dir,
            init_roi=shared_roi,
            tracker_factory=tracker_factory,
            skip_render=args.skip_render,
            step=args.step,
            use_ellipse_correction=args.ellipse_correct
        )

        if df is not None and vibration is not None:
            # 第一个视频的 ROI 复用于后续
            if shared_roi is None and used_roi is not None:
                shared_roi = used_roi
                print(f"\n[INFO] 将复用此 ROI 处理后续视频: {shared_roi}")

            all_summaries.append({
                "Video": video_name,
                "Frames": len(df),
                "Mode": "Ellipse+Tilt" if args.ellipse_correct else "Circle",
                "X_Amp_mm": vibration['amplitude_x_mm'],
                "Y_Amp_mm": vibration['amplitude_y_mm'],
                "Total_Amp_mm": vibration['amplitude_total_mm'],
                "X_Mean_mm": vibration['mean_x'] * 1000,
                "Y_Mean_mm": vibration['mean_y'] * 1000,
                "X_Std_mm": vibration['std_x'] * 1000,
                "Y_Std_mm": vibration['std_y'] * 1000,
            })

    # ---- 汇总表 ----
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(output_root, "summary_all_videos.xlsx")
        summary_df.to_excel(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"全部处理完成！")
        print(f"{'='*60}")
        print(f"总视频数: {len(all_summaries)}")
        print(f"汇总表: {summary_path}")
        print(f"\n振动幅度汇总 (mm):")
        print(summary_df.to_string(index=False))

        # 汇总柱状图
        _plot_summary_bar(summary_df, output_root)
    else:
        print("\n[WARN] 没有成功处理的视频")


def _plot_summary_bar(summary_df: pd.DataFrame, output_dir: str):
    """绘制所有视频的振动幅度对比柱状图。"""
    videos = summary_df["Video"].values
    x_amp = summary_df["X_Amp_mm"].values
    y_amp = summary_df["Y_Amp_mm"].values
    total_amp = summary_df["Total_Amp_mm"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vibration Amplitude Comparison Across Speeds", fontsize=14)

    x = np.arange(len(videos))
    width = 0.35

    # 子图1: X/Y 方向分组柱状图
    ax1 = axes[0]
    ax1.bar(x - width/2, x_amp, width, label='X Amplitude', color='steelblue')
    ax1.bar(x + width/2, y_amp, width, label='Y Amplitude', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(videos, rotation=45)
    ax1.set_ylabel('Amplitude (mm)')
    ax1.set_title('X/Y Direction Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2: 总振幅
    ax2 = axes[1]
    bars = ax2.bar(x, total_amp, width * 2, color='seagreen')
    ax2.set_xticks(x)
    ax2.set_xticklabels(videos, rotation=45)
    ax2.set_ylabel('Amplitude (mm)')
    ax2.set_title('Total Vibration Amplitude')
    ax2.grid(True, alpha=0.3, axis='y')
    # 标注数值
    for bar, val in zip(bars, total_amp):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "summary_comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] 汇总对比图已保存: {png_path}")


if __name__ == "__main__":
    main()
