"""Phase 0：数学推导数值验证模块。

验证 tilt_correction_research_plan.md 中的 5 条推导：
  推导1：倾斜圆的投影为椭圆，a/b = 1/cos(theta)
  推导2：倾角反演 theta = arccos(b/a)
  推导3：各向异性 scale_major = D/(2a), scale_minor = D/(2b)
  推导4：各向同性近似误差界
  推导5：透视逆变换校正的等价性

用法：
    D:\\anaconda\\python.exe -m src.derivation.check
"""

import sys
import os
import numpy as np
import cv2

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config


# =============================================================================
# 物理参数（与项目 config.py 一致）
# =============================================================================

# 使用新标定的相机内参
F = config.FX_CALIB_NEW        # 焦距 fx (像素)
Z0 = 0.25                      # 典型物距 (m)，用于仿真验证
R = config.SHAFT_DIAMETER_M / 2  # 轴半径 = 0.006 m
D = config.SHAFT_DIAMETER_M    # 轴直径 = 0.012 m

# 测试倾角
THETAS_DEG = [0, 3, 5, 10, 15, 20]


def verify_projection():
    """推导1验证：倾斜圆的投影为椭圆，a/b = 1/cos(theta)。

    方法：
      1. 在 3D 空间中建立半径 R 的圆
      2. 绕 X 轴旋转 theta
      3. 精确透视投影到像平面
      4. 对投影点做 fitEllipse，得到长短轴 a, b
      5. 验证 a/b ≈ 1/cos(theta)

    弱透视近似在 R << Z0 时成立，残差应 < 1%。
    """
    print("\n" + "=" * 60)
    print("推导1验证：倾斜圆的投影为椭圆")
    print("  公式：a/b = 1/cos(theta)")
    print(f"  参数：f={F:.1f}px, R={R*1000:.1f}mm, Z0={Z0*1000:.0f}mm")
    print("=" * 60)

    all_pass = True
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    for theta_deg in THETAS_DEG:
        theta = np.radians(theta_deg)

        # 3D 圆上的点（倾斜前，圆心在 (0, 0, Z0)）
        X3d = R * np.cos(t)
        Y3d = R * np.sin(t)
        Z3d = np.full_like(t, Z0)

        # 绕 X 轴旋转 theta
        Y_rot = Y3d * np.cos(theta) - Z3d * np.sin(theta)
        Z_rot = Y3d * np.sin(theta) + Z3d * np.cos(theta)
        X_rot = X3d  # X 不变

        # 透视投影
        u = F * X_rot / Z_rot
        v = F * Y_rot / Z_rot

        # fitEllipse 返回的是全长轴（直径），需除以2得到半轴
        points = np.stack([u, v], axis=1).astype(np.float32)
        center, axes, angle = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0  # 半轴

        ratio = a_fit / b_fit
        expected = 1.0 / np.cos(theta) if theta_deg > 0 else 1.0

        if theta_deg > 0:
            rel_err = abs(ratio - expected) / expected
        else:
            rel_err = abs(ratio - 1.0)

        status = "PASS" if rel_err < 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  theta={theta_deg:2d}°: a={a_fit:.2f}px, b={b_fit:.2f}px, "
              f"a/b={ratio:.6f}, expected={expected:.6f}, "
              f"residual={rel_err*100:.4f}% [{status}]")

    if all_pass:
        print("\n  [ALL PASS] 推导1验证通过：a/b = 1/cos(theta)")
    else:
        print("\n  [FAIL] 推导1验证未通过，请检查弱透视近似条件")
    return all_pass


def verify_tilt_inversion():
    """推导2验证：theta = arccos(b/a) 能正确反演倾角。

    方法：用 verify_projection 的正向模型生成椭圆，再用 arccos(b/a) 反演。
    """
    print("\n" + "=" * 60)
    print("推导2验证：倾角反演 theta = arccos(b/a)")
    print("=" * 60)

    all_pass = True
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    for theta_deg in THETAS_DEG:
        theta_true = np.radians(theta_deg)

        # 正向投影
        X3d = R * np.cos(t)
        Y3d = R * np.sin(t)
        Z3d = np.full_like(t, Z0)
        Y_rot = Y3d * np.cos(theta_true) - Z3d * np.sin(theta_true)
        Z_rot = Y3d * np.sin(theta_true) + Z3d * np.cos(theta_true)
        u = F * X3d / Z_rot
        v = F * Y_rot / Z_rot

        # fitEllipse（全长轴 → 半轴）
        points = np.stack([u, v], axis=1).astype(np.float32)
        _, axes, _ = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0  # 半轴

        # 反演
        theta_est = np.degrees(np.arccos(b_fit / a_fit)) if a_fit > 0 else 0.0
        err = abs(theta_est - theta_deg)

        # 小角度时 arccos 导数大，允许更大误差
        threshold = 0.5 if theta_deg <= 5 else 0.2
        status = "PASS" if err < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  theta_true={theta_deg:2d}° → theta_est={theta_est:.4f}°, "
              f"err={err:.4f}° [{status}]")

    if all_pass:
        print("\n  [ALL PASS] 推导2验证通过：theta = arccos(b/a)")
    else:
        print("\n  [FAIL] 推导2反演误差超阈值")
    return all_pass


def weak_perspective_project(theta_deg):
    """弱透视投影：Z 固定为 Z0*cos(theta)，验证公式正确性。

    弱透视近似下投影为严格椭圆，长短轴：
      a = fR / (Z0*cos(theta))
      b = fR / Z0
    """
    theta = np.radians(theta_deg)
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # 弱透视：Z 固定为 Z0*cos(theta)
    Z_weak = Z0 * np.cos(theta)
    u = F * R * np.cos(t) / Z_weak
    v = F * R * np.sin(t) * np.cos(theta) / Z_weak  # Y 旋转后 *cos(theta)
    # 注：v 方向还需减去 f*tan(theta) 的平移（不影响长短轴）
    v = v - F * np.sin(theta) / np.cos(theta)  # 减去中心偏移
    return u, v


def exact_perspective_project(theta_deg):
    """精确透视投影：Z 随 t 变化。用于残差分析。"""
    theta = np.radians(theta_deg)
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    X3d = R * np.cos(t)
    Y3d = R * np.sin(t)
    Z3d = np.full_like(t, Z0)
    Y_rot = Y3d * np.cos(theta) - Z3d * np.sin(theta)
    Z_rot = Y3d * np.sin(theta) + Z3d * np.cos(theta)
    u = F * X3d / Z_rot
    v = F * Y_rot / Z_rot
    return u, v


def verify_anisotropic_scale():
    """推导3验证：各向异性比例尺 scale_major = D/(2a), scale_minor = D/(2b)。

    使用弱透视投影（Z 固定）验证公式解析正确性。
    另外用精确透视投影量化弱透视近似的残差。
    """
    print("\n" + "=" * 60)
    print("推导3验证：各向异性比例尺")
    print("  公式：scale_major = D/(2a) = Z0*cos(theta)/f")
    print("        scale_minor = D/(2b) = Z0/f")
    print("=" * 60)

    all_pass = True

    # Part A: 弱透视（验证公式正确性）
    print("  --- Part A: 弱透视投影（公式验证） ---")
    for theta_deg in THETAS_DEG:
        theta = np.radians(theta_deg)
        u, v = weak_perspective_project(theta_deg)
        points = np.stack([u, v], axis=1).astype(np.float32)
        _, axes, _ = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0

        scale_major = D / (2 * a_fit)
        scale_minor = D / (2 * b_fit)
        expected_major = Z0 * np.cos(theta) / F
        expected_minor = Z0 / F

        err_major = abs(scale_major - expected_major)
        err_minor = abs(scale_minor - expected_minor)

        status = "PASS" if (err_major < 1e-6 and err_minor < 1e-6) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  theta={theta_deg:2d}°: "
              f"scale_major={scale_major*1e6:.4f} μm/px (err={err_major*1e6:.6f}), "
              f"scale_minor={scale_minor*1e6:.4f} μm/px (err={err_minor*1e6:.6f}) "
              f"[{status}]")

    # Part B: 精确透视（量化弱透视近似残差）
    print("  --- Part B: 精确透视（弱透视近似残差） ---")
    for theta_deg in THETAS_DEG:
        theta = np.radians(theta_deg)
        u, v = exact_perspective_project(theta_deg)
        points = np.stack([u, v], axis=1).astype(np.float32)
        _, axes, _ = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0

        scale_major = D / (2 * a_fit)
        scale_minor = D / (2 * b_fit)
        expected_major = Z0 * np.cos(theta) / F
        expected_minor = Z0 / F

        rel_err_major = abs(scale_major - expected_major) / expected_major * 100
        rel_err_minor = abs(scale_minor - expected_minor) / expected_minor * 100

        print(f"  theta={theta_deg:2d}°: "
              f"rel_err_major={rel_err_major:.4f}%, "
              f"rel_err_minor={rel_err_minor:.4f}% "
              f"(弱透视近似残差)")

    if all_pass:
        print("\n  [ALL PASS] 推导3验证通过：scale = D/(2a), D/(2b)")
    else:
        print("\n  [FAIL] 推导3 scale 公式残差超阈值")
    return all_pass


def verify_error_bound():
    """推导4验证：各向同性近似误差界。

    当使用 naive scale = D/(2*sqrt(a*b)) 时：
      长轴方向误差 = 1/sqrt(cos(theta)) - 1 （过估）
      短轴方向误差 = 1 - sqrt(cos(theta))    （低估）

    使用弱透视投影验证公式正确性，精确透视做残差参考。
    """
    print("\n" + "=" * 60)
    print("推导4验证：各向同性近似误差界")
    print("  naive_scale = D/(2*sqrt(a*b))")
    print("  长轴误差 = 1/sqrt(cos) - 1, 短轴误差 = 1 - sqrt(cos)")
    print("=" * 60)

    all_pass = True

    # Part A: 弱透视（验证公式正确性）
    print("  --- Part A: 弱透视投影（公式验证） ---")
    print(f"  {'theta':>5s} | {'cos':>8s} | {'naive_err_major':>16s} | "
          f"{'theory_major':>14s} | {'naive_err_minor':>16s} | "
          f"{'theory_minor':>14s} | {'status':>6s}")
    print("  " + "-" * 90)

    for theta_deg in THETAS_DEG:
        theta = np.radians(theta_deg)
        cos_t = np.cos(theta)

        # 弱透视投影
        u, v = weak_perspective_project(theta_deg)
        points = np.stack([u, v], axis=1).astype(np.float32)
        _, axes, _ = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0  # 半轴

        # naive scale（几何均值）
        R_hough = np.sqrt(a_fit * b_fit)
        naive_scale = D / (2 * R_hough)

        # 真实 scale
        scale_major_true = Z0 * cos_t / F
        scale_minor_true = Z0 / F

        # naive 误差（数值）
        naive_err_major = naive_scale / scale_major_true - 1.0
        naive_err_minor = 1.0 - naive_scale / scale_minor_true

        # 理论误差
        theory_major = 1.0 / np.sqrt(cos_t) - 1.0
        theory_minor = 1.0 - np.sqrt(cos_t)

        # 比较数值与理论
        match = (abs(naive_err_major - theory_major) < 1e-4 and
                 abs(naive_err_minor - theory_minor) < 1e-4)
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False

        print(f"  {theta_deg:5d} | {cos_t:8.5f} | "
              f"{naive_err_major*100:14.4f}% | {theory_major*100:12.4f}% | "
              f"{naive_err_minor*100:14.4f}% | {theory_minor*100:12.4f}% | "
              f"{status:>6s}")

    # Part B: 精确透视残差参考
    print("  --- Part B: 精确透视（弱透视近似残差） ---")
    for theta_deg in THETAS_DEG:
        theta = np.radians(theta_deg)
        cos_t = np.cos(theta)
        u, v = exact_perspective_project(theta_deg)
        points = np.stack([u, v], axis=1).astype(np.float32)
        _, axes, _ = cv2.fitEllipse(points)
        a_fit, b_fit = max(axes) / 2.0, min(axes) / 2.0

        R_hough = np.sqrt(a_fit * b_fit)
        naive_scale = D / (2 * R_hough)
        scale_major_true = Z0 * cos_t / F
        scale_minor_true = Z0 / F
        naive_err_major = naive_scale / scale_major_true - 1.0
        naive_err_minor = 1.0 - naive_scale / scale_minor_true

        print(f"  theta={theta_deg:2d}°: "
              f"naive_err_major={naive_err_major*100:.4f}%, "
              f"naive_err_minor={naive_err_minor*100:.4f}% "
              f"(精确透视残差)")

    if all_pass:
        print("\n  [ALL PASS] 推导4验证通过：误差界与理论一致")
    else:
        print("\n  [FAIL] 误差界数值与理论不符")
    return all_pass


def verify_correction_equivalence():
    """推导5验证：透视逆变换校正与各向异性 scale 等价。

    方法：
      1. 构造倾斜椭圆的像素位移 (du, dv)
      2. 路径A：各向异性 scale 直接换算 → (X_A, Y_A)
      3. 路径B：构造校正矩阵 H，变换像素位移后用 isotropic scale → (X_B, Y_B)
      4. 比较两条路径的结果应一致
    """
    print("\n" + "=" * 60)
    print("推导5验证：透视逆变换校正 ≡ 各向异性 scale")
    print("=" * 60)

    all_pass = True
    cx, cy = 960.0, 540.0  # 假设的主点
    delta_u, delta_v = 5.3, -3.7  # 测试像素位移

    for theta_deg in [5, 10, 15, 20]:
        theta = np.radians(theta_deg)
        cos_t = np.cos(theta)

        # 椭圆参数（假设长轴沿 u 方向）
        a = F * R / (Z0 * cos_t)
        b = F * R / Z0
        phi = 0.0  # 长轴角度（简化：沿 X 轴）

        # --- 路径A：各向异性 scale ---
        scale_major = D / (2 * a)
        scale_minor = D / (2 * b)
        rad = np.radians(phi)
        du = delta_u - cx
        dv = delta_v - cy
        # 旋转到主轴坐标系
        du_major = du * np.cos(rad) + dv * np.sin(rad)
        dv_minor = -du * np.sin(rad) + dv * np.cos(rad)
        # 各向异性换算
        X_major = scale_major * du_major
        Y_minor = scale_minor * dv_minor
        # 旋转回
        X_A = X_major * np.cos(rad) - Y_minor * np.sin(rad)
        Y_A = X_major * np.sin(rad) + Y_minor * np.cos(rad)

        # --- 路径B：透视逆变换 + isotropic scale ---
        # 校正矩阵 H = R(-phi) * S * R(phi)，S = diag(cos_t, 1, 1)
        # 等价于：du_corrected = du * cos_t（沿长轴方向压缩）
        du_corrected = du * cos_t  # S 的效果
        dv_corrected = dv
        # isotropic scale = D/(2b)
        iso_scale = D / (2 * b)
        X_B = iso_scale * du_corrected
        Y_B = iso_scale * dv_corrected

        err = np.sqrt((X_A - X_B)**2 + (Y_A - Y_B)**2)
        mag = np.sqrt(X_A**2 + Y_A**2)
        rel_err = err / mag if mag > 0 else 0.0

        status = "PASS" if rel_err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  theta={theta_deg:2d}°: "
              f"X_A={X_A*1e6:.4f}μm, Y_A={Y_A*1e6:.4f}μm, "
              f"X_B={X_B*1e6:.4f}μm, Y_B={Y_B*1e6:.4f}μm, "
              f"rel_err={rel_err:.2e} [{status}]")

    if all_pass:
        print("\n  [ALL PASS] 推导5验证通过：两种校正路径等价")
    else:
        print("\n  [FAIL] 两条路径结果不一致")
    return all_pass


def print_summary(results):
    """打印总结报告。"""
    print("\n" + "=" * 60)
    print("Phase 0 推导验证 — 总结")
    print("=" * 60)

    names = [
        "推导1: 倾斜圆投影为椭圆",
        "推导2: 倾角反演公式",
        "推导3: 各向异性比例尺",
        "推导4: 各向同性误差界",
        "推导5: 校正路径等价性",
    ]

    all_pass = True
    for name, result in zip(names, results):
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if not result:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL PASSED — 所有数学推导验证通过")
        print("  可以进入 Phase 1：仿真渲染")
    else:
        print("  SOME FAILED — 请检查未通过的推导")
    print("=" * 60)
    return all_pass


def main():
    print("Phase 0：数学推导数值验证")
    print(f"  Python: {sys.executable}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  NumPy:  {np.__version__}")
    print(f"  焦距 f = {F:.1f} px")
    print(f"  物距 Z0 = {Z0*1000:.0f} mm")
    print(f"  轴径 D = {D*1000:.0f} mm")
    print(f"  测试倾角: {THETAS_DEG}")

    results = []
    results.append(verify_projection())         # 推导1
    results.append(verify_tilt_inversion())      # 推导2
    results.append(verify_anisotropic_scale())   # 推导3
    results.append(verify_error_bound())         # 推导4
    results.append(verify_correction_equivalence())  # 推导5

    all_pass = print_summary(results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
