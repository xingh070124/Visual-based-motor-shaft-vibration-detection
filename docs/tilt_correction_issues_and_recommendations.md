# 电机轴倾斜补偿方案问题诊断与改进建议

> 项目：基于视觉的电机轴振动检测系统
> 日期：2026-07-09
> 作者：李昕泽 221001700218
> 审查范围：倾斜补偿方向的代码实现、验证流程、工程集成

---

## 问题总览

经对 `src/tracking/coordinate.py`、`src/tracking/shaft_detector.py`、`src/batch_analysis.py`、`test/sim/closure_validation.py`、`src/derivation/check.py` 及现有分析文档的逐行审查，识别出 10 个问题。

| 编号 | 类别 | 问题 | 严重度 | 影响范围 | 代码位置 |
|------|------|------|--------|----------|----------|
| 1 | 验证有效性 | 闭环验证为循环论证 | 致命 | Phase 7 验收结论不可信 | `test/sim/closure_validation.py:148-150` |
| 2 | 验证有效性 | 仿射变换旋转方向错误 | 高 | 图像校正功能不可用 | `test/sim/closure_validation.py:91` |
| 3 | 验证有效性 | 振动保持性测试用随机噪声替代真实变换 | 中 | 振动保持性结论无依据 | `test/sim/closure_validation.py:166-168` |
| 4 | 算法正确性 | 振动数据用 STARK bbox 中心而非椭圆中心 | 致命 | 倾斜补偿对位置测量无效 | `src/batch_analysis.py:535-536` |
| 5 | 算法正确性 | Scale 首帧固定 + 角度逐帧变化不一致 | 高 | 帧间噪声引入额外误差 | `src/batch_analysis.py:623-624,637` |
| 6 | 算法正确性 | 椭圆中心透视偏移未补偿 | 中 | 高精度场景下系统性偏移 | `src/tracking/coordinate.py:180-194` |
| 7 | 算法正确性 | 小角度退化未在生产代码中处理 | 中 | θ<3° 时倾角估计噪声放大 | `src/batch_analysis.py:365-377` |
| 8 | 工程集成 | 各向异性补偿改善量远低于理论预测 | 高 | 补偿方案的实际价值存疑 | `test/sim/results/eval_report.md` |
| 9 | 工程集成 | 仿真验证与生产管线参数不一致 | 中 | 验证结论无法迁移到生产 | `test/sim/closure_validation.py` vs `src/batch_analysis.py` |
| 10 | 工程集成 | 缺少真实数据 ground truth 验证 | 中 | 无法确认补偿在真实场景有效 | 全局 |

---

## 一、验证有效性问题

### 问题1：闭环验证为循环论证（致命）

**现状**

`test/sim/closure_validation.py` 的 `evaluate_closure()` 函数声称验证"用 θ_est 构造校正矩阵 → 校正图像 → 重新 fitEllipse → a'/b' 应趋近 1"。但实际代码（第148-150行）没有执行任何图像校正操作：

```python
# 实际代码：直接设 a_corrected = b，加随机噪声
noise_a = np.random.normal(1.0, 0.005)
noise_b = np.random.normal(1.0, 0.005)
a_corrected = b1 * noise_a  # 校正后长轴 = 短轴（加噪声）
b_corrected = b1 * noise_b  # 校正后短轴不变
ratio_after = a_corrected / b_corrected
```

这段代码将校正后的长轴直接设为短轴值 `b1`，然后检查 `a_corrected / b_corrected ≈ 1`。这在逻辑上等同于"设 a=b 后验证 a/b=1"，是同义反复，不构成对校正算法的任何检验。

`correct_ellipse_in_image()` 函数（第62-109行）虽然实现了完整的仿射变换逻辑，但在 `evaluate_closure()` 中从未被调用。函数定义后处于死代码状态。

**影响**

Phase 7 的验收报告（`closure_report.json`）中"补偿后 a'/b' → 1"的结论完全不可信。该结论不是来自算法验证，而是来自数学恒等式。基于此结论做出的"闭环验证通过"判断无效。

**建议**

重写 `evaluate_closure()`，执行真正的图像级闭环验证：

1. 在原始图像上检测椭圆，获取 `(a₁, b₁, angle₁, cx₁, cy₁)`
2. 修复 `correct_ellipse_in_image()` 的旋转方向问题（见问题2）
3. 对图像执行 `cv2.warpAffine` 校正
4. 在校正后图像上重新 `detect_ellipse_in_roi` + `cv2.fitEllipse`
5. 比较校正前后的 `a'/b'`，验证其趋近 1

---

### 问题2：仿射变换旋转方向错误（高）

**现状**

`correct_ellipse_in_image()` 第91行使用 `rad = math.radians(-angle)`，然后构造五个 3×3 矩阵 `M1`-`M5` 做链式乘法。`docs/tilt_solutions_analysis.md` 第82行记录了该问题的表现：θ=20° 时 a/b 从 1.064 恶化至 1.134，校正反而使椭圆度增大。

**根因**

`cv2.fitEllipse` 返回的 `angle` 是图像坐标系（Y 轴朝下）下椭圆长轴相对于水平方向的顺时针角度。而代码中的旋转矩阵 `M2`/`M4` 使用的是标准数学坐标系（Y 轴朝上）的旋转约定。两者方向相反，导致缩放方向偏离长轴，校正失效。

矩阵链 `M5 @ M4 @ M3 @ M2 @ M1` 中，`M2` 负责将长轴对齐到 X 轴，`M3` 在 X 方向做缩放，`M4` 旋转回。如果 `M2` 的旋转方向错误，`M3` 的缩放就不是沿长轴方向，而是沿某个偏离方向，导致椭圆被错误拉伸。

**影响**

图像校正功能完全不可用。由于 `correct_ellipse_in_image` 未被调用（见问题1），此 bug 目前不影响生产管线，但阻碍了 Phase 7 闭环验证的实施。

**建议**

用 `cv2.getRotationMatrix2D` 替代手动矩阵构造，减少方向错误风险。具体方法：

1. 以椭圆中心为锚点，用 `getRotationMatrix2D((cx, cy), angle, 1.0)` 将长轴对齐到水平方向
2. 在旋转后图像上沿 X 方向做缩放 `scale = b/a`
3. 再用 `getRotationMatrix2D((cx, cy), -angle, 1.0)` 旋转回
4. 添加 θ=0° 自洽性测试：输入正圆图像，校正后 a'/b' 应保持 ≈1.000（偏差 < 0.1%）

---

### 问题3：振动保持性测试用随机噪声替代真实变换（中）

**现状**

`evaluate_closure()` 第166-168行用随机数模拟校正后的中心偏移：

```python
center_noise = 0.05
du2 = du1 + np.random.normal(0, center_noise)
dv2 = dv1 + np.random.normal(0, center_noise)
```

振动保持率 `vib_preservation = disp_after / disp_before` 实际上是在计算 `(du + noise) / du`，其中 `noise` 是人为设定的 0.05px 高斯噪声。这个"保持率"与校正算法无关，只反映了随机数的统计特性。

**影响**

Phase 7 中"振动信号在补偿后保持不变"的结论没有实际验证支撑。无法确认各向异性换算是否会在某些方向上压缩振动信号。

**建议**

两种验证方式任选其一：

方式A（图像级）：在校正后图像上重新检测椭圆中心，比较校正前后中心坐标的差异。若差异 < 0.1px，则振动保持。

方式B（数学级，推荐）：构造已知像素位移 `(du, dv)`，用 `pixel_to_mm_anisotropic` 换算为物理位移，再与 ground truth 物理位移比较。验证不同方向（0°/45°/90°/135°）的换算结果，确认振动信号未被各向异性压缩。

---

## 二、算法正确性问题

### 问题4：振动数据用 STARK bbox 中心而非椭圆中心（致命）

**现状**

`src/batch_analysis.py` 第535-536行，振动位移取自 STARK 跟踪器的 bbox 几何中心：

```python
track_cx = pred_box[0] + pred_box[2] / 2
track_cy = pred_box[1] + pred_box[3] / 2
```

在椭圆模式下（第540行），椭圆检测返回的中心坐标被丢弃：

```python
_, _, vis_a, vis_b, vis_angle = detect_ellipse_in_roi(
    frame, pred_box, ..., return_params=True
)
```

返回值的第一、二个元素（椭圆中心 cx, cy）被赋给 `_`，只有半轴和角度被保留。

**影响**

这是倾斜补偿方案中最根本的设计缺陷。各向异性 scale 只修正了像素到物理量的换算系数，但被换算的像素坐标本身（STARK bbox 中心）并未经过倾斜校正。

在精确透视投影下，空间圆心的投影点与拟合椭圆的几何中心不重合。偏移量近似为：

$$\Delta c \approx \frac{f R^2 \sin\theta}{2 Z_0^2 \cos\theta}$$

代入项目参数（f=2928.5px, R=6mm, Z₀=250mm, θ=20°）：

$$\Delta c \approx \frac{2928.5 \times 36 \times 10^{-6} \times 0.342}{2 \times 0.0625 \times 0.940} \approx 0.034 \text{ px}$$

换算为物理位移：0.034px × 60μm/px ≈ 2.0μm。对于 50-100μm 级的振动幅度，这是 2-4% 的系统性偏移。

更关键的是，STARK bbox 中心与椭圆中心在每帧的偏差方向不同（取决于 bbox 如何框住目标），引入的是非系统性噪声而非固定偏移，无法通过差分消除。

**建议**

在椭圆模式下，用椭圆检测中心替代 STARK bbox 中心作为振动数据源：

```python
ell_cx, ell_cy, vis_a, vis_b, vis_angle = detect_ellipse_in_roi(...)
# 使用椭圆中心而非 bbox 中心
track_cx = ell_cx
track_cy = ell_cy
```

同时保留 STARK bbox 作为 ROI 定位器（为椭圆检测提供搜索区域），但不将其中心用于振动测量。这样倾斜补偿同时作用于位置测量和 scale 换算，形成完整的校正链路。

---

### 问题5：Scale 首帧固定 + 角度逐帧变化不一致（高）

**现状**

`src/batch_analysis.py` 中，各向异性 scale 从首帧椭圆参数一次性计算后固定不变（第623-624行）：

```python
scale_major = config.SHAFT_DIAMETER_M / (2.0 * init_a)
scale_minor = config.SHAFT_DIAMETER_M / (2.0 * init_b)
```

但在逐帧坐标换算时（第637行），使用的是每帧检测的椭圆角度：

```python
cam_x, cam_y = pixel_to_mm_anisotropic(
    px, py, config.CX, config.CY,
    scale_major, scale_minor, ell_angle  # ell_angle 来自逐帧检测
)
```

`ell_angle` 是 `row[5]`，即每帧 `detect_ellipse_in_roi` 返回的角度值。

**影响**

如果电机轴在运行中倾角不变（合理假设），那么 `a`、`b`、`angle` 三个参数在物理上应保持恒定。但由于检测噪声，逐帧 `ell_angle` 会有波动（评估报告显示 θ_err std 约 0.1-0.6°）。当角度波动而 scale 不变时，旋转-缩放-反旋转的操作会把缩放施加到错误的方向上，引入额外的坐标误差。

具体地，设真实角度 φ，检测角度 φ+δ。旋转到主轴坐标系时，位移分量在 major/minor 方向的分配发生偏差 δ，导致本应沿长轴方向的位移被部分分配到短轴方向（或反之），乘以不同的 scale 后产生误差。

**建议**

采用方案A（首帧固定）：scale 和角度都使用首帧检测值 `init_a`、`init_b`、`init_angle`。这与项目"首帧标定"的整体理念一致（scale 本身就是首帧标定的）。逐帧检测的椭圆参数仅用于可视化和半径历史更新，不参与坐标换算。

```python
# 坐标换算时用首帧固定角度
cam_x, cam_y = pixel_to_mm_anisotropic(
    px, py, config.CX, config.CY,
    scale_major, scale_minor, init_angle  # 改为首帧角度
)
```

---

### 问题6：椭圆中心透视偏移未补偿（中）

**现状**

`docs/tilt_solutions_analysis.md` 第19行列出了"椭圆中心偏移"作为三类系统性误差之一，但在代码实现中未做补偿。`pixel_to_mm_anisotropic`（`src/tracking/coordinate.py:180-194`）直接使用像素坐标减去主点作为位移，没有加入透视偏移修正项。

**量化分析**

偏移量公式（弱透视一阶近似）：

$$\Delta v_{\text{offset}} = \frac{f R^2 \sin\theta}{2 Z_0^2 \cos\theta}$$

| θ (°) | 偏移量 (px) | 物理偏移 (μm) | 占 50μm 振幅比例 |
|-------|------------|--------------|----------------|
| 0 | 0 | 0 | 0% |
| 5 | 0.008 | 0.5 | 1.0% |
| 10 | 0.017 | 1.0 | 2.0% |
| 15 | 0.026 | 1.5 | 3.1% |
| 20 | 0.034 | 2.0 | 4.1% |

在 θ=20° 时，透视偏移导致 2.0μm 的系统性误差，占 50μm 振幅的 4.1%。但由于该偏移是直流分量（固定方向），对峰峰值振幅（max-min）的影响取决于振动方向与偏移方向的关系——如果振动沿偏移方向，峰峰值不受影响；如果垂直，则完全不受影响。实际影响通常小于 1%。

**建议**

在当前精度要求下（振幅 50-100μm，误差容忍 ~5%），此偏移可暂不补偿。但在论文中应明确讨论这一误差源及其量级，说明为何在当前参数下可忽略。若未来精度要求提升到 <1%，则需在 `pixel_to_mm_anisotropic` 中加入修正项。

---

### 问题7：小角度退化未在生产代码中处理（中）

**现状**

`test/sim/results/eval_report.md` 显示 θ=0° 时 `θ_err mean=1.668°`、`θ_err max=3.015°`。根因是 `arccos(b/a)` 在 `b/a → 1` 时导数发散，0.5% 的椭圆检测噪声被放大到 ±2-3° 的倾角波动。

文档 `docs/tilt_solutions_analysis.md` 第406行建议"设 θ < 3° 时不补偿（退化处理）"，但 `src/batch_analysis.py` 的 `--ellipse-correct` 模式始终应用各向异性 scale，未做任何退化判断（第365-377行）。

**影响**

当实际倾角接近 0°（电机轴与相机近似垂直）时，`--ellipse-correct` 模式反而会引入额外误差。检测噪声导致每帧 θ_est 在 0-3° 之间随机波动，各向异性 scale 随之波动，把本应稳定的 scale 因子变成了噪声源。

在 θ=3° 时，naive scale 的理论误差仅 0.14%（见 `docs/tilt_solutions_analysis.md` 第342行），远小于检测噪声引入的误差。此时不补偿反而更准确。

**建议**

在 `process_single_video` 中加入倾角阈值判断：

```python
if use_ellipse_correction:
    init_theta = estimate_tilt_angle(init_a, init_b)
    if init_theta < 3.0:
        # 退化处理：倾角过小，回退到各向同性 scale
        print(f"  [ELLIPSE] θ_est={init_theta:.2f}° < 3°, 退化为各向同性 scale")
        use_ellipse_correction = False
        # 走圆形检测 + 各向同性换算路径
```

---

## 三、工程集成问题

### 问题8：各向异性补偿改善量远低于理论预测（高）

**现状**

`test/sim/results/eval_report.md` 第50行记录了各向异性 vs naive 的振幅误差对比：

| θ_true (°) | 各向异性 mean (%) | naive mean (%) | 实际差异 | 理论差异 |
|-----------|-----------------|---------------|---------|---------|
| 0 | 7.42 | 7.41 | 0.01% | 0% |
| 10 | 9.32 | 9.45 | 0.13% | 0.77% |
| 15 | 10.03 | 10.27 | 0.24% | 1.75% |
| 20 | 10.40 | 10.82 | 0.42% | 3.16% |

实际改善量（0.42%）仅为理论预测（3.16%）的 13%。报告将主误差归因于 fitEllipse 中心精度（~0.1px → 1.7% 底噪），但这无法解释 0.42% vs 3.16% 的巨大差距——如果中心噪声是主要误差源，它应该同时影响各向异性和 naive 两种模式，而 scale 误差的差异应该仍然体现为理论值 3.16%。

**可能原因**

1. **仿真倾角与标称值不一致**：θ=20° 的仿真图像实际椭圆度 a/b=1.064，但 1/cos(20°)=1.064 理论上应该精确匹配。问题可能在于评估脚本提取的 a/b 是 fitEllipse 的含噪结果，而非 ground truth 值。如果检测噪声使 a/b 波动 ±2%，那么 scale_major/scale_minor 的比值也随之波动，补偿方向错误时反而增大误差。

2. **坐标换算用 fitEllipse 中心而非 GT 中心**：如果评估脚本用检测中心做坐标换算，中心检测噪声（~0.1px → 1.7%）会淹没 scale 改善（0.77-3.16%）。需要分离评估 scale 误差和中心检测误差。

3. **scale 计算用含噪 a/b**：各向异性 scale 直接从 fitEllipse 的 a/b 计算，检测噪声导致 scale 本身有误差。如果 a 偏大 1%，scale_major 偏小 1%，补偿效果被削弱。

**建议**

1. 在仿真评估中分离误差来源：用 GT 中心坐标做坐标换算（消除中心噪声），单独验证各向异性 scale 的理论改善量是否为 3.16%
2. 检查仿真渲染的实际倾角：对渲染图像做 fitEllipse，用多帧平均的 a/b 反算 θ，与渲染参数 θ_true 对比
3. 如果分离后 scale 改善仍远低于理论值，需检查 `pixel_to_mm_anisotropic` 的旋转矩阵实现是否正确

---

### 问题9：仿真验证与生产管线参数不一致（中）

**现状**

| 参数 | `closure_validation.py` | `batch_analysis.py`（椭圆模式） |
|------|------------------------|-------------------------------|
| circularity_threshold | 默认值 0.3 | 0.3（首帧）/ 0.85（圆形模式） |
| ROI 构造 | 固定方形 ROI（基于主点 CX/CY） | STARK bbox 动态更新 |
| 振动数据源 | 椭圆检测中心 | STARK bbox 中心 |
| 坐标换算 | 不涉及（仅验证椭圆度） | `pixel_to_mm_anisotropic` |
| 预期半径 | `(D/2 * F) / Z0`（Z-based） | `estimate_expected_radius_from_roi`（ROI-based） |

**影响**

仿真验证中椭圆检测的调用方式与生产管线不同。验证结论（如"椭圆拟合成功率 100%"、"倾角反演精度 < 1°"）是在固定 ROI + 低圆度阈值的条件下得到的，不能直接推断生产管线中 STARK bbox 动态 ROI 下的表现。

特别是，生产管线中 STARK bbox 的尺寸和位置会逐帧变化，可能超出椭圆检测的有效范围，导致检测成功率下降。

**建议**

在仿真验证中复现生产管线的完整流程：STARK 跟踪器初始化 → 逐帧跟踪获取 bbox → 在 bbox 内做椭圆检测 → 各向异性换算。或者至少统一检测参数和 ROI 构造方式，使验证结果具有可迁移性。

---

### 问题10：缺少真实数据 ground truth 验证（中）

**现状**

所有验证均在 `test/sim/` 下的仿真数据完成。`docs/tilt_correction_research_plan.md` 第753-755行的论文章节映射中将"真实数据 sanity check"列为后续工作。`test/video/` 中的 10-100 RPM 视频已在 `10.mp4` 上做过圆形 vs 椭圆模式对比（commit `92d4497`），但缺少 ground truth 倾角，无法判断哪种模式更准确。

实际电机轴与相机之间的倾角完全未知。如果实际倾角 < 3°，则倾斜补偿不仅无效反而有害（见问题7）；如果实际倾角 > 10°，则补偿有显著价值。

**建议**

用方案J（标定板预标定）获取真实倾角 ground truth：

1. 在电机轴端面位置放置平面标定板（棋盘格）
2. 用已标定的相机拍摄，通过 `cv2.solvePnP` 求解标定板法向量
3. 法向量与光轴的夹角即为系统倾角 θ₀
4. 用 θ₀ 验证 `estimate_tilt_angle` 的反演精度

如果 θ₀ < 3°，说明实验台机械对准良好，倾斜补偿可不做（退化处理）。如果 θ₀ > 5°，说明需要补偿，且可量化补偿前后振幅测量的差异。

---

## 四、改进建议优先级

按修复紧迫程度排列：

### P0 — 致命（立即修复）

| 编号 | 问题 | 修复内容 | 预期效果 |
|------|------|---------|---------|
| 1 | 闭环验证循环论证 | 重写 `evaluate_closure()`，接入 `correct_ellipse_in_image()` 做真实图像校正 | Phase 7 验收结论可信 |
| 4 | 振动数据用 bbox 中心 | 椭圆模式下用椭圆中心替代 bbox 中心 | 倾斜补偿对位置测量生效 |

### P1 — 高（尽快修复）

| 编号 | 问题 | 修复内容 | 预期效果 |
|------|------|---------|---------|
| 2 | 仿射变换旋转方向 | 用 `getRotationMatrix2D` 重写，添加 θ=0° 自洽测试 | 图像校正功能可用 |
| 5 | Scale/角度不一致 | 坐标换算中用首帧固定角度 `init_angle` | 消除帧间角度噪声 |
| 8 | 补偿改善量异常 | 分离 scale 误差和中心检测误差评估 | 确认补偿方案有效性 |

### P2 — 中（后续完善）

| 编号 | 问题 | 修复内容 | 预期效果 |
|------|------|---------|---------|
| 3 | 振动保持性测试 | 用已知像素位移做数学级验证 | 振动保持性结论有依据 |
| 6 | 透视偏移未补偿 | 量化偏移量级，论文中讨论 | 明确误差边界 |
| 7 | 小角度退化 | 加入 θ<3° 退化判断 | 避免小角度噪声放大 |
| 9 | 仿真/生产参数不一致 | 统一检测参数或复现完整流程 | 验证结论可迁移 |
| 10 | 缺少真实数据验证 | 标定板预标定获取 θ₀ | 确认补偿在真实场景有效 |

---

## 五、总结

当前倾斜补偿方案的数学推导（推导1-5）经数值验证全部通过，理论框架成立。但在工程实现层面存在两个致命问题：

1. **闭环验证是循环论证**，Phase 7 的"验证通过"结论无效，整个验证链路需要重做
2. **振动数据使用 STARK bbox 中心而非椭圆中心**，导致倾斜补偿只修正了 scale 系数但未修正位置测量，补偿链路不完整

此外，各向异性补偿的实际改善量（0.42%）远低于理论预测（3.16%），需要分离误差来源重新评估，确认补偿方案是否真正有效。

建议按 P0 → P1 → P2 的顺序修复，在 P0 问题解决前，不宜将当前倾斜补偿方案用于论文的实验结果章节。
