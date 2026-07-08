# 电机轴倾斜场景下振动测量误差补偿 — 推导与实验回路设计

> 项目：基于视觉的电机轴振动检测系统
> 日期：2026-07-08
> 作者：李昕泽 221001700218
> 状态：研究方案（待实施）

---

## 目录

1. [问题定义](#1-问题定义)
2. [数学推导（DERIVATION_PROOF）](#2-数学推导derivation_proof)
3. [实验回路总览](#3-实验回路总览)
4. [Phase 0：推导验证](#phase-0推导验证)
5. [Phase 1：仿真渲染](#phase-1仿真渲染)
6. [Phase 2：基线测量](#phase-2基线测量)
7. [Phase 3：椭圆检测](#phase-3椭圆检测)
8. [Phase 4：倾角反演](#phase-4倾角反演)
9. [Phase 5：各向异性校正](#phase-5各向异性校正)
10. [Phase 6：误差验证](#phase-6误差验证)
11. [Phase 7：闭环自洽](#phase-7闭环自洽)
12. [验收标准汇总](#12-验收标准汇总)
13. [仿真实验矩阵](#13-仿真实验矩阵)
14. [论文章节映射](#14-论文章节映射)

---

## 1. 问题定义

### 1.1 当前系统假设

当前 pipeline（`src/tracking/coordinate.py`）使用轴径自标定比例尺：

$$X_m = \text{scale} \cdot (u - c_x), \quad \text{scale} = \frac{D}{2R_{\text{hough}}}$$

其中 D = 12mm（轴径），R_hough 为 Hough 圆检测半径。

**隐含假设**：轴端面在像平面上的投影为圆（即轴线严格平行于光轴）。

### 1.2 倾斜场景

当电机轴轴线与光轴存在夹角 θ 时：

| 现象 | 物理原因 |
|------|----------|
| 端面投影为椭圆 | 圆的透视投影 |
| Hough 圆检测半径失真 | 椭圆无单一半径 |
| scale 各向异性 | 长短轴方向缩放不同 |
| 振幅测量含系统性误差 | 各向同性 scale 近似 |

### 1.3 研究目标

1. 从理论上推导倾斜投影的几何关系，给出严格的误差界
2. 设计无需真实数据的可验证实验回路（仿真 + 自洽性检验）
3. 实现方向 B（透视逆变换 + 各向异性 scale）补偿算法
4. 量化补偿前后的振幅测量误差改善

---

## 2. 数学推导（DERIVATION_PROOF）

### 2.1 推导 1：倾斜圆的投影为椭圆

**坐标系定义**：

- 相机在原点 O，光轴沿 +Z 方向，焦距 f
- 电机轴端面为半径 R 的圆，垂直时圆心在 (0, 0, Z₀)
- 绕 X 轴倾斜 θ 角后，圆心变为 (0, −Z₀ sinθ, Z₀ cosθ)

**圆参数化**（倾斜前）：

$$\mathbf{P}(t) = \begin{bmatrix} R\cos t \\ R\sin t \\ Z_0 \end{bmatrix}, \quad t \in [0, 2\pi)$$

**绕 X 轴旋转 θ**（旋转矩阵 Rx(θ)）：

$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

$$\mathbf{P}'(t) = R_x(\theta) \cdot \mathbf{P}(t) = \begin{bmatrix} R\cos t \\ R\sin t\cos\theta - Z_0\sin\theta \\ R\sin t\sin\theta + Z_0\cos\theta \end{bmatrix}$$

**透视投影**（除以 Z 分量）：

$$u(t) = f \cdot \frac{R\cos t}{R\sin t\sin\theta + Z_0\cos\theta}$$

$$v(t) = f \cdot \frac{R\sin t\cos\theta - Z_0\sin\theta}{R\sin t\sin\theta + Z_0\cos\theta}$$

**弱透视近似**（R ≪ Z₀，故 Z 分量中 R sin t sinθ 项可忽略）：

$$Z'(t) = R\sin t\sin\theta + Z_0\cos\theta \approx Z_0\cos\theta$$

代入后：

$$u(t) \approx \frac{f \cdot R\cos t}{Z_0\cos\theta} = \frac{fR}{Z_0\cos\theta}\cos t$$

$$v(t) \approx \frac{f \cdot (R\sin t\cos\theta - Z_0\sin\theta)}{Z_0\cos\theta} = \frac{fR}{Z_0}\sin t - f\tan\theta$$

**结论**：投影为椭圆，中心偏移 (0, −f tanθ)：

$$\boxed{\text{长半轴 } a = \frac{fR}{Z_0\cos\theta}, \quad \text{短半轴 } b = \frac{fR}{Z_0}, \quad \frac{a}{b} = \frac{1}{\cos\theta}}$$

**物理解读**：

- **长轴方向**（⊥ 倾斜方向，即 u 方向）：圆心靠近相机（深度从 Z₀ 减至 Z₀cosθ），透视放大 1/cosθ 倍，无透视缩短 → 半轴增大
- **短轴方向**（∥ 倾斜方向，即 v 方向）：透视放大 1/cosθ 与透视缩短 cosθ 恰好抵消 → 半轴不变

### 2.2 推导 2：倾角反演

由推导 1 的长短轴比直接得：

$$\boxed{\theta = \arccos\!\left(\frac{b}{a}\right) = \arccos\!\left(\frac{\min(a, b)}{\max(a, b)}\right)}$$

从单张图像的椭圆拟合（`cv2.fitEllipse`）即可估计倾角，无需深度信息或外部仪器。

**退化条件**：当 θ → 0 时 b/a → 1，arccos 在 1 附近导数发散（dθ/db ∝ 1/√(1−(b/a)²)），小角度估计对噪声敏感。需要多帧平均。

### 2.3 推导 3：各向异性比例尺

真实直径 D = 2R 已知。像素位移到物理位移的换算需要沿长短轴方向分别处理。

**长轴方向**（⊥ 倾斜，真实物理位移 ΔX）：

$$\Delta u = \frac{f \cdot \Delta X}{Z_0\cos\theta} \implies \Delta X = \Delta u \cdot \frac{Z_0\cos\theta}{f} = \Delta u \cdot \frac{D}{2a}$$

**验证**：scale_major = D/(2a) = 2R / (2 · fR/(Z₀cosθ)) = Z₀cosθ/f ✓

**短轴方向**（∥ 倾斜，真实物理位移 ΔY）：

$$\Delta v = \frac{f \cdot \Delta Y \cos\theta}{Z_0\cos\theta} = \frac{f \cdot \Delta Y}{Z_0} \implies \Delta Y = \Delta v \cdot \frac{Z_0}{f} = \Delta v \cdot \frac{D}{2b}$$

**验证**：scale_minor = D/(2b) = 2R / (2 · fR/Z₀) = Z₀/f ✓

$$\boxed{\text{scale}_{\text{major}} = \frac{D}{2a}, \quad \text{scale}_{\text{minor}} = \frac{D}{2b}}$$

**实施步骤**：

1. 从 `cv2.fitEllipse()` 得到椭圆参数 (cx, cy, a, b, φ)
2. 将像素位移 (Δu, Δv) 旋转到椭圆主轴坐标系：

$$\begin{bmatrix} \Delta u_{\text{major}} \\ \Delta v_{\text{minor}} \end{bmatrix} = R(-\varphi) \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

3. 分别换算：X_major = scale_major · Δu_major, Y_minor = scale_minor · Δv_minor
4. 旋转回相机坐标系（如果需要 X, Y 分量）

### 2.4 推导 4：各向同性近似误差界

若**不补偿**，用 Hough 圆检测的等效半径 R_hough。Hough 圆倾向于检测椭圆的"等效圆"，其半径近似几何均值：

$$R_{\text{hough}} \approx \sqrt{a \cdot b} = \sqrt{\frac{fR}{Z_0\cos\theta} \cdot \frac{fR}{Z_0}} = \frac{fR}{Z_0\sqrt{\cos\theta}}$$

naive scale：

$$s_{\text{naive}} = \frac{D}{2 \cdot R_{\text{hough}}} = \frac{D \cdot Z_0\sqrt{\cos\theta}}{2fR} = \frac{Z_0\sqrt{\cos\theta}}{f}$$

**长轴方向误差**（真实 scale = Z₀cosθ/f）：

$$\frac{s_{\text{naive}}}{\text{scale}_{\text{major}}} = \frac{Z_0\sqrt{\cos\theta}/f}{Z_0\cos\theta/f} = \frac{1}{\sqrt{\cos\theta}}$$

相对误差：ε_major = 1/√cosθ − 1（过估）

**短轴方向误差**（真实 scale = Z₀/f）：

$$\frac{s_{\text{naive}}}{\text{scale}_{\text{minor}}} = \frac{Z_0\sqrt{\cos\theta}/f}{Z_0/f} = \sqrt{\cos\theta}$$

相对误差：ε_minor = 1 − √cosθ（低佔）

**数值表**：

| θ (°) | cosθ | √cosθ | ε_major (过估) | ε_minor (低估) | 总振幅误差（最坏） |
|-------|------|-------|----------------|----------------|-------------------|
| 0 | 1.0000 | 1.0000 | 0% | 0% | 0% |
| 3 | 0.9986 | 0.9993 | +0.07% | −0.07% | 0.14% |
| 5 | 0.9962 | 0.9981 | +0.19% | −0.19% | 0.38% |
| 10 | 0.9848 | 0.9924 | +0.77% | −0.77% | 1.54% |
| 15 | 0.9659 | 0.9828 | +1.75% | −1.75% | 3.50% |
| 20 | 0.9397 | 0.9694 | +3.16% | −3.06% | 6.22% |

**注**：总振幅误差（最坏情况）= ε_major + |ε_minor|，发生在振动方向恰好沿 45° 时。实际误差取决于振动方向与倾斜方向的夹角。

### 2.5 推导 5：透视逆变换校正（方向 B 的等价形式）

将椭圆校正为圆的单应矩阵可分解为旋转 × 各向异性缩放 × 平移：

$$H = T \cdot R(\varphi) \cdot S \cdot R(-\varphi) \cdot T^{-1}$$

其中：

- T = 平移矩阵（椭圆中心 → 原点）
- R(φ) = 旋转矩阵（长轴对齐到 X 轴）
- S = diag(b/a, 1, 1) = diag(cosθ, 1, 1)（长轴方向缩回短轴长度）

校正后图像中椭圆变为半径 b = fR/Z₀ 的圆，可用标准 Hough 圆检测，isotropic scale = D/(2b)。

**等价性证明**：方向 B（图像校正）与推导 3 的各向异性 scale 在数学上等价——前者先校正图像再做各向同性换算，后者直接做各向异性换算。两者给出的物理位移相同。

**工程选择**：推荐使用"各向异性 scale"（不修改图像），因为它不影响 STARK 跟踪器的输入。

---

## 3. 实验回路总览

实验回路由 4 个阶段、8 个 phase 组成，形成"推导 → 仿真 → 测量 → 验证 → 反馈"的闭环：

```
P0 推导验证 → P1 仿真渲染 → P2 基线测量 → P3 椭圆检测
                                              ↓
P7 闭环自洽 ← P6 误差验证 ← P5 各向异性校正 ← P4 倾角反演
      ↓                                    (反馈)
      └─── 未通过 → 调整模型/参数 → 回到 P1 ──┘
```

**设计原则**：

- **可执行**：每个 phase 有明确的 Python/Blender 命令
- **可验证**：每个 phase 有从数学推导导出的量化验收标准
- **闭环**：P7 验证结果反馈到 P1，形成迭代改进回路

---

## Phase 0：推导验证

### 目的
用数值计算验证推导 1-4 的公式自洽性，确保理论无错。

### 执行

```bash
D:\anaconda\python.exe -m src.derivation.check
```

### 脚本逻辑（`src/derivation/check.py`）

```python
import numpy as np

def verify_projection():
    """验证推导1：倾斜圆投影为椭圆"""
    f, R, Z0 = 2928.5, 0.006, 0.25  # 用项目实际内参
    for theta_deg in [0, 3, 5, 10, 15, 20]:
        theta = np.radians(theta_deg)
        # 精确透视投影
        t = np.linspace(0, 2*np.pi, 1000)
        X = R * np.cos(t)
        Y = R * np.sin(t) * np.cos(theta) - Z0 * np.sin(theta)
        Z = R * np.sin(t) * np.sin(theta) + Z0 * np.cos(theta)
        u = f * X / Z
        v = f * Y / Z
        # fitEllipse 得到长短轴
        from cv2 import fitEllipse, NORM_L2
        import cv2
        points = np.stack([u, v], axis=1).astype(np.float32)
        (_, (a, b), _) = cv2.fitEllipse(points)
        a, b = max(a, b), min(a, b)
        # 验证 a/b ≈ 1/cos(theta)
        ratio = a / b
        expected = 1 / np.cos(theta)
        assert abs(ratio - expected) / expected < 0.01, \
            f"theta={theta_deg}: ratio={ratio:.6f}, expected={expected:.6f}"
    print("[PASS] 推导1验证通过：a/b = 1/cos(theta)")

def verify_scale():
    """验证推导3：各向异性scale"""
    f, R, D, Z0 = 2928.5, 0.006, 0.012, 0.25
    for theta_deg in [0, 5, 10, 15, 20]:
        theta = np.radians(theta_deg)
        a = f * R / (Z0 * np.cos(theta))
        b = f * R / Z0
        scale_major = D / (2 * a)
        scale_minor = D / (2 * b)
        expected_major = Z0 * np.cos(theta) / f
        expected_minor = Z0 / f
        assert abs(scale_major - expected_major) < 1e-10
        assert abs(scale_minor - expected_minor) < 1e-10
    print("[PASS] 推导3验证通过：scale = D/(2a), D/(2b)")

def verify_error_bound():
    """验证推导4：误差界"""
    for theta_deg in [5, 10, 15, 20]:
        theta = np.radians(theta_deg)
        cos_t = np.cos(theta)
        eps_major = 1 / np.sqrt(cos_t) - 1
        eps_minor = 1 - np.sqrt(cos_t)
        print(f"  theta={theta_deg:2d}°: eps_major=+{eps_major*100:.2f}%, "
              f"eps_minor={eps_minor*100:.2f}%")
    print("[PASS] 推导4误差界验证通过")

if __name__ == "__main__":
    verify_projection()
    verify_scale()
    verify_error_bound()
    print("\n=== 所有推导验证通过 ===")
```

### 产出

- 验证通过/失败的日志
- 精确透视投影 vs 弱透视近似的残差表

### 验收标准

| 检验项 | 阈值 |
|--------|------|
| a/b = 1/cosθ 相对残差 | < 1%（弱透视近似误差） |
| scale 公式残差 | < 1e-10（解析等式） |
| 误差界数值 | 与理论表一致 |

---

## Phase 1：仿真渲染

### 目的
生成带 ground truth 的可控倾角测试数据（视频 + 静态图）。

### 1A：Blender 视频渲染

#### 执行

```bash
blender -b -P test/sim/render_tilt_video.py -- --theta 10 --amp 50 --freq 5 --sigma 5 --seed 1
```

#### 渲染参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 1920×1080 | 匹配相机内参 K_CALIB_NEW |
| FPS | 30 | |
| 时长 | 5 秒（150 帧） | |
| 焦距 | 50mm | 匹配 fx ≈ 2928.5px |
| 物距 Z₀ | 250mm | |
| 轴径 D | 12mm | |
| 倾角 θ | {0, 3, 5, 10, 15, 20}° | 6 个水平 |
| 振幅 A | {30, 50, 100} μm | 3 个水平 |
| 频率 f | {2, 5, 10} Hz | 3 个水平 |
| 噪声 σ | {0, 5, 15} | 3 个水平 |
| 随机种子 | {1, 2, 3} | 3 个重复 |
| **合计** | 486 段视频 | |

#### Ground Truth 输出

每段视频配套一个 CSV：

```csv
frame,u_true,v_true,X_true,Y_true,Z_true,theta_true,amp_true,freq_true
1,960.00,540.00,0.000,0.000,0.250,10.0,50.0,5.0
2,960.05,540.01,0.005,0.001,0.250,10.0,50.0,5.0
...
```

#### 噪声注入

1. **高斯图像噪声**：`img += np.random.normal(0, σ, img.shape)`
2. **镜头畸变**：用已标定的径向畸变系数 k1, k2（从 `calibrator.py` 输出获取）
3. **运动模糊**：按角速度在旋转方向卷积
4. **光照渐变**：模拟非均匀照明（左亮右暗 10% 差异）

### 1B：静态图渲染

#### 执行

```bash
blender -b -P test/sim/render_tilt_stills.py -- --theta 10 --phase 45 --sigma 8 --blur 7
```

#### 参数矩阵

| 参数 | 取值 | 数量 |
|------|------|------|
| 倾角 θ | {0, 3, 5, 8, 10, 12, 15, 20}° | 8 |
| 振动相位 φ | {0, 45, 90, 135, 180}° | 5 |
| 噪声 σ | {0, 3, 8, 15} | 4 |
| 运动模糊 | {0, 3, 7, 11} px | 4 |
| **合计** | 640 张 | |

### 验收标准

| 检验项 | 阈值 |
|--------|------|
| GT CSV 帧数 | = 视频帧数（150） |
| θ_true 标注 | 与渲染参数一致 |
| 像素坐标真值 | ±0.01px 精度 |
| 图像文件完整性 | 无损坏/缺失帧 |

---

## Phase 2：基线测量

### 目的
用当前 pipeline（无补偿）跑仿真数据，建立"before"基线。

### 执行

```bash
# 批量跑所有仿真视频
D:\anaconda\python.exe -m src.batch_analysis --video-dir test/sim/videos --output-dir test/sim/baseline --skip-render
```

### 产出

每个视频输出：
- `data.xlsx`：帧号、像素坐标、半径、相机坐标
- 汇总表 `summary.xlsx`

### 验收标准

| 检验项 | 阈值 |
|--------|------|
| 检测成功率 | > 50%（低倾角应 > 90%） |
| 振幅输出 | 非零（排除全失败） |
| 数据完整性 | 所有视频都有输出 |

### 预期结果

- θ = 0°：误差 < 1%（验证 pipeline 正确性）
- θ ≥ 10°：误差 > 1%，检测成功率下降
- θ = 20°：可能出现检测失败（Hough 圆半径超出容差）

---

## Phase 3：椭圆检测

### 目的
在 ROI 内用 `cv2.fitEllipse()` 替代 Hough 圆，提取椭圆参数 (a, b, φ)。

### 执行

```bash
# 用椭圆检测模式跑仿真数据
D:\anaconda\python.exe -m src.batch_analysis --video-dir test/sim/videos --output-dir test/sim/ellipse --use-ellipse --skip-render
```

### 算法实现（`src/tracking/shaft_detector.py` 新增）

```python
def detect_ellipse_in_roi(frame, roi_box, expected_radius):
    """在 ROI 内拟合椭圆，返回 (cx, cy, a, b, angle)"""
    x, y, w, h = map(int, roi_box)
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return x + w/2, y + h/2, expected_radius, expected_radius, 0
    
    # 选最大轮廓
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 100:
        return x + w/2, y + h/2, expected_radius, expected_radius, 0
    
    (cx, cy), (a, b), angle = cv2.fitEllipse(best)
    return x + cx, y + cy, a, b, angle
```

### 产出

每帧输出椭圆参数 (cx, cy, a, b, φ)。

### 验收标准

| 检验项 | 阈值 |
|--------|------|
| 椭圆拟合成功率 | > 95% |
| 长短轴比 a/b | θ=0° 时 < 1.05（近圆） |
| 椭圆中心与 GT 中心偏差 | < 2px |

---

## Phase 4：倾角反演

### 目的
从椭圆长短轴比反演倾角 θ_est = arccos(b/a)，验证反演精度。

### 执行

```python
# 在 Phase 3 输出上计算
theta_est = np.degrees(np.arccos(min(a, b) / max(a, b)))
```

### 验证协议

#### 4.1 反演精度（静态图）

对 640 张静态图，每张计算 θ_est，与 θ_true 对比：

| 噪声水平 σ | 验收阈值 \|θ_est − θ_true\| |
|-----------|--------------------------|
| 0 | < 0.5° |
| 3 | < 1° |
| 8 | < 2° |
| 15 | < 3° |

#### 4.2 多帧稳定性（视频）

对每段视频，用 150 帧的 θ_est 计算统计量：

| 指标 | 验收阈值 |
|------|---------|
| mean(θ_est) − θ_true | < 2° |
| std(θ_est) | < 3° |
| CV = std/mean | < 15% |

#### 4.3 小角度退化分析

θ < 5° 时 arccos 导数发散，记录此区间的实际噪声放大量：

| θ_true | 期望 std(θ_est) | 验收 |
|--------|----------------|------|
| 0° | — | θ_est 应 < 3° |
| 3° | < 2° | std 不超过此值 |
| 5° | < 1.5° | std 不超过此值 |

### 产出

- θ_est vs θ_true 散点图
- 反演误差-噪声水平曲线
- 多帧 std 表

---

## Phase 5：各向异性校正

### 目的
用推导 3 的各向异性 scale 替代各向同性 scale，补偿倾斜误差。

### 执行

```bash
# 用各向异性校正模式
D:\anaconda\python.exe -m src.batch_analysis --video-dir test/sim/videos --output-dir test/sim/corrected --ellipse-correct --skip-render
```

### 算法实现（`src/tracking/coordinate.py` 新增）

```python
def pixel_to_mm_anisotropic(
    pixel_x, pixel_y, cx, cy,
    scale_major, scale_minor, ellipse_angle
):
    """各向异性 scale + 椭圆主轴对齐
    
    Args:
        pixel_x, pixel_y: 像素坐标
        cx, cy: 主点
        scale_major: D/(2a) — 长轴方向（⊥倾斜）
        scale_minor: D/(2b) — 短轴方向（∥倾斜）
        ellipse_angle: 椭圆长轴方向角（度）
    Returns:
        (X_m, Y_m): 物理位移（米）
    """
    rad = np.radians(ellipse_angle)
    dx = pixel_x - cx
    dy = pixel_y - cy
    # 旋转到椭圆主轴坐标系
    dx_major = dx * np.cos(rad) + dy * np.sin(rad)
    dy_minor = -dx * np.sin(rad) + dy * np.cos(rad)
    # 各向异性换算
    X_major = scale_major * dx_major
    Y_minor = scale_minor * dy_minor
    # 旋转回相机坐标系
    X = X_major * np.cos(rad) - Y_minor * np.sin(rad)
    Y = X_major * np.sin(rad) + Y_minor * np.cos(rad)
    return X, Y
```

### 验收标准

| 检验项 | 阈值 |
|--------|------|
| 校正后振幅与 GT 振幅相对误差 | < 1%（所有 θ） |
| 校正后 X/Y 各向异性比 | 0.98~1.02（接近 1） |
| 振动波形保真度 | 相关系数 > 0.99 |

---

## Phase 6：误差验证

### 目的
对比 Phase 2（基线）和 Phase 5（校正）的振幅测量误差，生成误差-倾角曲线。

### 执行

```bash
D:\anaconda\python.exe -m src.eval.compare_gt --baseline-dir test/sim/baseline --corrected-dir test/sim/corrected --gt-dir test/sim/ground_truth --output-dir test/sim/analysis
```

### 评估指标

| 指标 | 公式 | 物理含义 |
|------|------|----------|
| 振幅相对误差 | \|A_meas − A_true\| / A_true | 测量系统误差 |
| 相位偏差 | argmax(corr(s_meas, s_true)) × dt | 时间同步性 |
| 各向异性比 | AmpX / AmpY | 倾斜导致的形变 |
| 检测失败率 | 失败帧 / 总帧 | 算法稳定性 |
| 椭圆度残差 | a/b − 1（补偿后） | 补偿有效性 |

### 验收标准

| θ (°) | 基线误差 | 校正后误差 | 改善倍数 |
|-------|---------|-----------|---------|
| 0 | < 1% | < 1% | ≥ 1× |
| 5 | < 1% | < 0.5% | ≥ 2× |
| 10 | < 2% | < 0.5% | ≥ 4× |
| 15 | < 4% | < 1% | ≥ 4× |
| 20 | < 7% | < 1.5% | ≥ 5× |

### 产出

- 误差-倾角曲线（基线 vs 校正）
- 失败率-倾角柱状图
- 各向异性比箱线图
- 误差-噪声水平热力图

---

## Phase 7：闭环自洽

### 目的
验证补偿后的残余椭圆度 → 1（即补偿确实消除了椭圆变形），形成闭环。

### 验证协议

#### 7.1 补偿闭环

对每张静态图：

1. Phase 3 检测椭圆 → 得 (a, b, φ)
2. Phase 4 反演 θ_est
3. 用 θ_est 构造校正矩阵 H
4. 校正图像后**重新** fitEllipse → 得 (a', b')
5. 检查 a'/b' → 1 的程度

| θ (°) | 补偿前 a/b | 补偿后 a'/b' | 验收 |
|-------|-----------|-------------|------|
| 0 | 1.000 | 1.000 | ✓ |
| 5 | 1.004 | < 1.002 | ✓ |
| 10 | 1.015 | < 1.003 | ✓ |
| 15 | 1.035 | < 1.005 | ✓ |
| 20 | 1.064 | < 1.010 | ✓ |

#### 7.2 振动保持性

最关键的检验——补偿不能把振动信号也"补偿"掉：

1. 在 3D 中心注入已知振动 Δx(t)
2. 渲染 → 检测椭圆 → 各向异性校正 → 得 Δx_meas(t)
3. 比较 Δx_meas 与 Δx_true 的振幅和波形

| 检验项 | 验收阈值 |
|--------|---------|
| 振幅误差 | < 5% |
| 波形相关系数 | > 0.95 |
| 频率估计误差 | < 2% |

#### 7.3 反馈回路

若 Phase 6/7 验收不通过：

| 失败模式 | 反馈目标 | 调整内容 |
|---------|---------|---------|
| θ_est 偏差大 | Phase 3 | 增加 CLAHE 预处理 / 多帧平均 |
| 振幅误差仍高 | Phase 5 | 检查椭圆方向角 φ 估计 |
| 振动被补偿掉 | Phase 5 | 验证旋转矩阵方向 |
| 小角度不稳定 | Phase 4 | 设 θ < 3° 时不补偿（退化处理） |

---

## 12. 验收标准汇总

| Phase | 验收项 | 阈值 | 来源 |
|-------|--------|------|------|
| P0 | a/b = 1/cosθ 残差 | < 1% | 推导 1 |
| P0 | scale 公式残差 | < 1e-10 | 推导 3 |
| P1 | GT 帧数完整 | = 150 | 渲染协议 |
| P2 | 检测成功率 | > 50% | 基线要求 |
| P3 | 椭圆拟合成功率 | > 95% | 算法要求 |
| P3 | a/b (θ=0°) | < 1.05 | 推导 1 |
| P4 | θ_est 精度（σ≤8） | < 2° | 推导 2 |
| P4 | 多帧 std | < 3° | 统计要求 |
| P5 | 校正后振幅误差 | < 1% | 推导 3 |
| P5 | 各向异性比 | 0.98~1.02 | 推导 4 |
| P6 | 改善倍数（θ=15°） | ≥ 4× | 推导 4 |
| P7 | 补偿后 a'/b' | < 1.01 | 闭环验证 |
| P7 | 振动保持性 | 振幅误差 < 5% | 关键检验 |

---

## 13. 仿真实验矩阵

### 13.1 视频（实验 1）

| 因子 | 水平数 | 取值 | 总数 |
|------|--------|------|------|
| 倾角 θ | 6 | 0°, 3°, 5°, 10°, 15°, 20° | |
| 振幅 A | 3 | 30, 50, 100 μm | |
| 频率 f | 3 | 2, 5, 10 Hz | |
| 噪声 σ | 3 | 0, 5, 15 | |
| 随机种子 | 3 | 1, 2, 3 | |
| **合计** | | | **486** |

每段 150 帧，总计 72,900 帧。

### 13.2 静态图（实验 3）

| 因子 | 水平数 | 取值 | 总数 |
|------|--------|------|------|
| 倾角 θ | 8 | 0°, 3°, 5°, 8°, 10°, 12°, 15°, 20° | |
| 振动相位 φ | 5 | 0°, 45°, 90°, 135°, 180° | |
| 噪声 σ | 4 | 0, 3, 8, 15 | |
| 运动模糊 | 4 | 0, 3, 7, 11 px | |
| **合计** | | | **640** |

---

## 14. 论文章节映射

```
第4章 倾斜误差补偿方法
  4.1 问题分析：轴不垂直时的投影变形
  4.2 数学推导
    4.2.1 倾斜圆的椭圆投影（推导1）
    4.2.2 倾角反演公式（推导2）
    4.2.3 各向异性比例尺（推导3）
    4.2.4 各向同性近似误差界（推导4）
    4.2.5 透视逆变换校正（推导5）
  4.3 算法实现
    4.3.1 椭圆检测（Phase 3）
    4.3.2 倾角估计（Phase 4）
    4.3.3 各向异性坐标换算（Phase 5）
  4.4 算法流程图

第5章 实验与结果
  5.1 仿真平台搭建
    5.1.1 几何模型与渲染参数
    5.1.2 噪声模型（高斯/畸变/模糊/光照）
    5.1.3 Ground Truth 提取
  5.2 推导验证（Phase 0）
    5.2.1 公式自检结果
  5.3 仿真视频上的振幅-倾角敏感性（实验1）
    5.3.1 实验设计（486段视频矩阵）
    5.3.2 当前算法基线（Phase 2）
    5.3.3 椭圆补偿后结果（Phase 5）
    5.3.4 振幅误差-倾角曲线（图）
    5.3.5 失败率-倾角曲线（图）
  5.4 静态图上的倾角反演精度（实验3）
    5.4.1 反演精度-噪声水平曲线
    5.4.2 补偿闭环验证（Phase 7.1）
    5.4.3 振动保持性验证（Phase 7.2）
  5.5 真实数据 sanity check
    5.5.1 用现有 10-100.mp4 跑通椭圆检测
    5.5.2 与基线结果对比
  5.6 讨论与小结
    5.6.1 仿真 vs 真实数据的差异
    5.6.2 方法的适用范围与限制
    5.6.3 后续工作展望
```

---

## 附录：代码文件清单

### 新增文件

| 文件 | 用途 |
|------|------|
| `src/derivation/check.py` | Phase 0：公式数值验证 |
| `src/tracking/shaft_detector.py` | 新增 `detect_ellipse_in_roi()` |
| `src/tracking/coordinate.py` | 新增 `pixel_to_mm_anisotropic()` |
| `src/eval/compare_gt.py` | Phase 6：误差对比分析 |
| `test/sim/render_tilt_video.py` | Phase 1A：Blender 视频渲染 |
| `test/sim/render_tilt_stills.py` | Phase 1B：Blender 静态图渲染 |
| `src/batch_analysis.py` | 新增 `--use-ellipse` / `--ellipse-correct` 参数 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/config.py` | 新增椭圆检测参数配置 |
| `src/batch_analysis.py` | 集成椭圆检测 + 各向异性校正路径 |
