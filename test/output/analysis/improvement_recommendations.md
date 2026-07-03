# 电机轴圆形检测不稳定性 — 改进建议报告

## 改进1：移除半径自适应正反馈（P0 — 致命修复）

### 问题
`batch_analysis.py:478-480` 的 `expected_radius = vis_radius` 每帧更新，
形成正反馈崩溃：小半径检测 → expected缩小 → Hough范围缩小 → 只能检到更小圆。

### 当前代码
```python
# batch_analysis.py:478-480
if vis_radius > 0:
    expected_radius = vis_radius
```

### 建议修改 — 方案B（滑动窗口中位数，推荐）
```python
from collections import deque

# 在循环前初始化：
radius_history = deque(maxlen=30)

# 循环中替换原来的自适应更新：
if vis_radius > 0:
    radius_history.append(vis_radius)
if len(radius_history) >= 10:
    expected_radius = float(np.median(list(radius_history)))
```

### 验证指标
- 重跑视频90/100：Radius列稳定在90-110px区间
- zero_radius_rate 降为 0%
- scale 偏差 < 5%

---

## 改进2：收紧 Hough 检测半径范围（P1 — 高优先级）

### 问题
`shaft_detector.py:93-94` 的范围 `[0.4*expected, 1.8*expected]` 过宽，
允许检测到轴承外圈等非目标圆形特征。

### 当前代码
```python
# shaft_detector.py:93-94
minRadius=int(expected_radius_pixels * 0.4),
maxRadius=int(expected_radius_pixels * 1.8)
```

### 建议修改
```python
minRadius=int(expected_radius_pixels * 0.85),  # ±15%
maxRadius=int(expected_radius_pixels * 1.15)
```

### 验证指标
- 重跑视频30/60/80：Radius分布从双峰变为单峰（~100px）
- scale 偏差从 -17~-19% 降至 < 5%

---

## 改进3：提高霍夫 param2 + 统一预处理（P1 — 高优先级）

### 问题
1. `param2=25` 过低，容易产生假阳性圆
2. 策略1(霍夫)在灰度图上直接运行，策略2(轮廓)先做高斯模糊 → 预处理不一致

### 当前代码
```python
# shaft_detector.py:87-95
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 策略1: 霍夫圆检测（无模糊！）
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
    param1=100, param2=25, ...)
```

### 建议修改
```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, blur_kernel, 0)  # 统一预处理

# 策略1: 霍夫圆检测（用模糊图）
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1,
    minDist=int(expected_radius_pixels * 1.5),  # 根据预期半径设置最小间距
    param1=100, param2=40,  # 25→40 减少假阳性
    minRadius=int(expected_radius_pixels * 0.85),
    maxRadius=int(expected_radius_pixels * 1.15)
)
```

### 验证指标
- 全视频半径CV%降低（目标 < 5%）
- 无假阳性圆检测

---

## 改进4：比例尺标定改用首帧或全局中位数（P0 — 致命修复）

### 问题
`batch_analysis.py:508-515` 取单个视频前50帧有效半径中位数标定 scale。
当半径已开始崩溃时（视频90/100），中位数被污染。

### 当前代码
```python
# batch_analysis.py:508-515
radii = np.array([r[3] for r in results])
valid = radii[radii > 0]
N_calib = min(config.SCALE_CALIB_FRAMES, len(valid))
calib = valid[:N_calib]
median_r = float(np.median(calib))
scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, median_r)
```

### 建议修改 — 方案A（首帧标定，最简单可靠）
```python
# 首帧通常最清晰、无运动模糊
# 在循环前用首帧检测半径标定 scale
if init_radius > 0:
    scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, init_radius)
else:
    # 回退：用ROI估算
    est_r = estimate_expected_radius_from_roi(used_roi)
    scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, est_r)
```

### 建议修改 — 方案B（全局标定，跨视频中位数）
```python
# 收集全部视频全部帧的有效半径
all_valid_radii = []  # 在批量处理中收集
global_r_median = np.median(all_valid_radii)
scale = compute_scale_m_per_px(config.SHAFT_DIAMETER_M, global_r_median)
```

### 验证指标
- 全部视频 scale 接近 60 μm/px（偏差 < 3%）
- 视频90/100振幅从1.8+mm降至 ~1.1mm
- 视频30/60/80振幅修正增大约20%

---

## 改进5：添加时域滤波（P2 — 中优先级）

### 问题
STARK bbox中心直接用作振动数据，无任何时域滤波，高频抖动明显。

### 当前代码
```python
# batch_analysis.py:465-467
track_cx = pred_box[0] + pred_box[2] / 2
track_cy = pred_box[1] + pred_box[3] / 2
```

### 建议修改 — 后处理滑动平均
```python
# 在全部帧处理完成后，对坐标序列做滑动平均
window = 5
px_seq = np.array([r[1] for r in results])
py_seq = np.array([r[2] for r in results])

# 边缘保持的滑动平均
px_smooth = np.convolve(px_seq, np.ones(window)/window, mode='same')
py_smooth = np.convolve(py_seq, np.ones(window)/window, mode='same')

# 用平滑后的坐标换算
for i, row in enumerate(results):
    cam_x, cam_y = pixel_to_mm(px_smooth[i], py_smooth[i], config.CX, config.CY, scale)
    row.append(cam_x)
    row.append(cam_y)
```

### 验证指标
- Cam_X/Y 标准差降低 30-50%
- 振动曲线更平滑，但峰峰值保持不变

---

## 改进6：振幅计算去漂移（P2 — 中优先级）

### 问题
`analyzer.py:44` 的 `amplitude = max - min` 包含缓慢漂移（热膨胀/轴承磨合），
导致振幅偏大。

### 当前代码
```python
# analyzer.py:44
amplitude_x = float(np.max(cam_x) - np.min(cam_x))
```

### 建议修改 — 增加去趋势选项
```python
from scipy import signal as sig

def compute_vibration_amplitude(cam_x, cam_y, detrend=False):
    if detrend:
        cam_x = sig.detrend(cam_x)
        cam_y = sig.detrend(cam_y)

    amplitude_x = float(np.max(cam_x) - np.min(cam_x))
    amplitude_y = float(np.max(cam_y) - np.min(cam_y))
    # ... 其余不变
```

### 验证指标
- 视频50的X方向漂移（0.4mm）不再计入振幅
- 去趋势后振幅降低 ~10-20%（取决于漂移量）

---

## 实施优先级与预期效果

| 优先级 | 改进 | 预期效果 | 实施难度 |
|--------|------|---------|---------|
| P0 | 改进1：移除半径正反馈 | 视频90/100检测恢复正常 | 低（改3行代码） |
| P0 | 改进4：比例尺首帧标定 | 全部视频振幅准确 | 低（改5行代码） |
| P1 | 改进2：收紧Hough范围 | 视频80误检消除 | 低（改2行参数） |
| P1 | 改进3：提高param2+统一预处理 | 全部视频假阳性减少 | 低（改3行代码） |
| P2 | 改进5：时域滤波 | 高频噪声降低30-50% | 中（加后处理） |
| P2 | 改进6：去漂移振幅 | 振幅更准确 | 低（加1个参数） |

### 建议实施顺序
1. 先实施改进1+4（P0），重跑视频90/100验证
2. 再实施改进2+3（P1），重跑视频80验证
3. 最后实施改进5+6（P2），全量重跑验证
