"""Phase 1: 仿真渲染脚本 — 纯 Python 实现。

由于环境无 Blender 和 bpy，采用基于推导1（精确透视投影）的解析渲染：
  - 根据倾斜角 theta 精确投影圆为椭圆
  - 添加振动位移、噪声、运动模糊、光照渐变
  - 输出 PNG 静态图 + ground_truth CSV

用法：
    python -m test.sim.render_tilted_stills
    或：
    D:\\anaconda\\python.exe test/sim/render_tilted_stills.py
"""

import os
import sys
import csv
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config


# =============================================================================
# 物理参数（与项目 config.py 一致）
# =============================================================================

F = config.FX_CALIB_NEW        # 焦距 fx ≈ 2928.5 px
CX = config.CX_CALIB_NEW        # 主点 (1013, 757)
CY = config.CY_CALIB_NEW
Z0 = 0.25                      # 物距 (m)
D = config.SHAFT_DIAMETER_M    # 轴径 0.012 m
R = D / 2                      # 轴半径

# 图像尺寸（基于主点位置调整）
IMG_W = int(2 * CX)
IMG_H = int(2 * CY)

# 椭圆颜色（灰度，模拟金属轴端面）
INTENSITY_BG = 30              # 背景灰度
INTENSITY_AXIS = 180           # 轴端面灰度


# =============================================================================
# 精确透视投影
# =============================================================================

def project_tilted_circle(theta_deg, vibration_x=0.0, vibration_y=0.0, n_points=360):
    """精确透视投影：3D 圆 → 2D 椭圆点集。

    Args:
        theta_deg: 倾角（度）
        vibration_x: X 方向振动位移 (m)
        vibration_y: Y 方向振动位移 (m)
        n_points: 采样点数

    Returns:
        u, v: 像平面坐标 (n_points,)
    """
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    X3 = R * np.cos(t) + vibration_x
    Y3 = R * np.sin(t) + vibration_y
    Z3 = np.full_like(t, Z0)

    # 绕 X 轴旋转 theta
    Y_rot = Y3 * cos_t - Z3 * sin_t
    Z_rot = Y3 * sin_t + Z3 * cos_t

    # 透视投影
    u = F * X3 / Z_rot + CX
    v = F * Y_rot / Z_rot + CY

    # 消除倾斜导致的中心偏移（f*tan(theta) 在 v 方向）
    # 真实场景中相机瞄准轴端面中心，所以图像中心就是端面中心
    # 减去这个偏移让椭圆中心始终在 (CX, CY)
    v = v + F * sin_t / cos_t  # 减去 -f*tan(theta) = 加 f*tan(theta)

    return u, v


def draw_ellipse_pixels(u, v, shape=(IMG_H, IMG_W), intensity=INTENSITY_AXIS):
    """根据椭圆点集生成光栅化图像。

    流程：
      1. 用 u, v 创建 mask
      2. PIL 多边形填充
      3. 抗锯齿
    """
    h, w = shape
    img = Image.new('L', (w, h), INTENSITY_BG)
    draw = ImageDraw.Draw(img)

    points = list(zip(u.tolist(), v.tolist()))
    if len(points) >= 3:
        draw.polygon(points, fill=intensity)

    return np.array(img, dtype=np.float32)


def add_gradient_lighting(img, direction='horizontal', strength=0.15):
    """添加光照渐变（模拟非均匀照明）。"""
    h, w = img.shape
    if direction == 'horizontal':
        gradient = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)
        img = img * gradient[np.newaxis, :]
    else:
        gradient = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)
        img = img * gradient[:, np.newaxis]
    return np.clip(img, 0, 255)


def add_gaussian_noise(img, sigma):
    """添加高斯噪声。"""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)


def add_motion_blur(img, length=0):
    """添加运动模糊（各向同性 box filter）。"""
    if length <= 0:
        return img
    import cv2
    kernel = np.ones((length, length), dtype=np.float32) / (length * length)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


# =============================================================================
# 渲染器
# =============================================================================

def render_static(
    theta_deg,
    vibration_phase=0.0,
    sigma=5.0,
    blur_len=0,
    amp_um=50.0,
    seed=1,
    out_path=None,
    gt_path=None
):
    """渲染单张静态图（含 ground truth）。

    Args:
        theta_deg: 倾角（度）
        vibration_phase: 振动相位（度）
        sigma: 高斯噪声标准差（灰度）
        blur_len: 运动模糊核长度（像素，0 表示不加）
        amp_um: 振动振幅（微米）
        seed: 随机种子
        out_path: PNG 输出路径
        gt_path: GT CSV 输出路径
    """
    np.random.seed(seed)

    # 计算振动位移
    amp_m = amp_um * 1e-6
    vx = amp_m * math.cos(math.radians(vibration_phase))
    vy = amp_m * math.sin(math.radians(vibration_phase))

    # 精确透视投影获取椭圆点集
    u, v = project_tilted_circle(theta_deg, vibration_x=vx, vibration_y=vy)

    # 渲染
    img = draw_ellipse_pixels(u, v)
    img = add_gradient_lighting(img, 'horizontal', 0.15)
    img = add_gaussian_noise(img, sigma)
    img = add_motion_blur(img, blur_len)

    # 保存图像
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(img.astype(np.uint8)).save(out_path)

    # 计算 GT 椭圆参数
    import cv2
    points = np.stack([u, v], axis=1).astype(np.float32)
    (cx_gt, cy_gt), (w_gt, h_gt), ang_gt = cv2.fitEllipse(points)
    a_gt, b_gt = max(w_gt, h_gt) / 2.0, min(w_gt, h_gt) / 2.0
    theta_est_gt = math.degrees(math.acos(min(b_gt, a_gt) / max(b_gt, a_gt)))

    # 保存 GT
    if gt_path:
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        with open(gt_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'param', 'value'
            ])
            writer.writerow(['theta_true_deg', theta_deg])
            writer.writerow(['vibration_phase_deg', vibration_phase])
            writer.writerow(['amp_um', amp_um])
            writer.writerow(['sigma', sigma])
            writer.writerow(['blur_len', blur_len])
            writer.writerow(['seed', seed])
            writer.writerow(['ellipse_cx_px', cx_gt])
            writer.writerow(['ellipse_cy_px', cy_gt])
            writer.writerow(['ellipse_a_px', a_gt])
            writer.writerow(['ellipse_b_px', b_gt])
            writer.writerow(['ellipse_angle_deg', ang_gt])
            writer.writerow(['vibration_x_m', vx])
            writer.writerow(['vibration_y_m', vy])
            writer.writerow(['theta_est_arcmin', theta_est_gt])
            writer.writerow(['vibration_x_um', vx * 1e6])
            writer.writerow(['vibration_y_um', vy * 1e6])

    return {
        'theta_true': theta_deg,
        'theta_est': theta_est_gt,
        'cx': cx_gt, 'cy': cy_gt,
        'a': a_gt, 'b': b_gt,
        'angle': ang_gt,
        'vx_um': vx * 1e6,
        'vy_um': vy * 1e6,
    }


# =============================================================================
# 批量生成
# =============================================================================

def batch_render():
    """批量生成静态图（Phase 1 协议）。"""
    output_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'images')
    gt_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'ground_truth')

    # 实验矩阵（按方案文档）
    thetas = [0, 3, 5, 8, 10, 12, 15, 20]            # 8 个
    phases = [0, 45, 90, 135, 180]                    # 5 个
    sigmas = [0, 3, 8, 15]                            # 4 个
    blurs = [0, 3, 7, 11]                             # 4 个
    seeds = [1]                                       # 1 个（控制总数量）

    total = len(thetas) * len(phases) * len(sigmas) * len(blurs) * len(seeds)
    print(f"开始渲染 {total} 张静态图...")
    print(f"  倾角: {thetas}")
    print(f"  相位: {phases}")
    print(f"  噪声: {sigmas}")
    print(f"  模糊: {blurs}")

    results = []
    count = 0
    for theta in thetas:
        for phase in phases:
            for sigma in sigmas:
                for blur in blurs:
                    for seed in seeds:
                        count += 1
                        name = (
                            f"theta{theta:02d}_phase{phase:03d}"
                            f"_s{sigma:02d}_b{blur:02d}_r{seed}"
                        )
                        out_png = os.path.join(output_dir, f"{name}.png")
                        out_csv = os.path.join(gt_dir, f"{name}.csv")

                        result = render_static(
                            theta_deg=theta,
                            vibration_phase=phase,
                            sigma=sigma,
                            blur_len=blur,
                            amp_um=500.0,  # 500μm — 投影到像素约 5.86px，超出 fitEllipse 分辨率
                            seed=seed,
                            out_path=out_png,
                            gt_path=out_csv,
                        )
                        results.append(result)
                        if count % 50 == 0 or count == total:
                            print(f"  [{count:4d}/{total}] {name}.png")

    # 汇总
    print(f"\n完成 {count} 张静态图渲染")
    print(f"图像: {output_dir}")
    print(f"GT:   {gt_dir}")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: 倾斜轴仿真渲染'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速测试模式：只渲染 theta=0,10 几个图'
    )
    args = parser.parse_args()

    if args.quick:
        # 快速验证：2 倾角 × 2 噪声
        output_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'images')
        gt_dir = os.path.join(_PROJECT_ROOT, 'test', 'sim', 'ground_truth')
        for theta in [0, 10]:
            for sigma in [0, 5]:
                name = f"quick_theta{theta:02d}_s{sigma:02d}"
                render_static(
                    theta_deg=theta,
                    vibration_phase=0,
                    sigma=sigma,
                    blur_len=0,
                    amp_um=50.0,
                    seed=1,
                    out_path=os.path.join(output_dir, f"{name}.png"),
                    gt_path=os.path.join(gt_dir, f"{name}.csv"),
                )
                print(f"  渲染: {name}.png")
        return

    batch_render()


if __name__ == "__main__":
    main()
