# Fig. 1 — System Pipeline Diagram 制作提示词

> 目标：IEEE TIM 双栏论文，宽度 ≈ 8.5 cm（单栏），白底矢量图，线宽 ≥ 1pt，字体 ≥ 8pt
> 推荐工具：TikZ（最精确）> draw.io > Inkscape > Python matplotlib

---

## 图片结构：5 个阶段水平/垂直排列

```
┌──────────────────────────────────────────────────────────┐
│                     Stage 1: 相机标定                      │
│   ┌─────────┐                                            │
│   │ 棋盘格图像 │ → 张氏标定 → K = [fx 0 cx; 0 fy cy; 0 0 1] │
│   │ (22张)   │                                            │
│   └─────────┘                                            │
├──────────────────────────────────────────────────────────┤
│                     Stage 2: 目标初始化                    │
│   ┌─────────┐     ┌──────────┐                           │
│   │ 首帧图像  │  →  │ 手动选 ROI │ → 椭圆拟合 → (cx,cy,a,b,φ) │
│   └─────────┘     └──────────┘                           │
├──────────────────────────────────────────────────────────┤
│                     Stage 3: STARK 跟踪                    │
│   ┌─────────┐     ┌──────────────────┐                   │
│   │ 第 t 帧  │  →  │ STARK Transformer │ → bbox [x,y,w,h]  │
│   └─────────┘     └──────────────────┘                   │
├──────────────────────────────────────────────────────────┤
│                     Stage 4: 椭圆精检测                    │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌─────────┐             │
│   │ CLAHE │→│ Canny │→│ Hough │→│fitEllipse│ → (u,v,a,b) │
│   └──────┘  └──────┘  └──────┘  └─────────┘             │
├──────────────────────────────────────────────────────────┤
│                     Stage 5: 比例尺 + 振动                 │
│   ┌──────────────┐     ┌────────────────┐                │
│   │ s = D·a/(2b²) │  →  │ X = s·Δu       │ → A_X = max-min│
│   │ θ = arccos(b/a)│     │ Y = s·Δv       │                │
│   └──────────────┘     └────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

---

## 配色方案

| 元素 | 颜色 | HEX |
|------|------|-----|
| 阶段框背景 | 浅蓝 | `#E3F2FD` |
| 阶段框边框 | 深蓝 | `#1565C0` |
| 数据节点 | 白底灰边 | `#FAFAFA` / `#757575` |
| 箭头 | 深灰 | `#424242` |
| 核心公式 | 深红强调 | `#C62828` |
| 文字 | 黑色 | `#212121` |

---

## 详细规格

### Stage 1: 相机标定
- 左侧：一组 9×6 棋盘格图标（2×3 小方格即可，不必画全 9×6），标注 "22 images"
- 箭头 → 标有 "Zhang's method" 的框
- 右侧输出：矩阵 K 的 LaTeX 渲染（`K = [2928.5, 0, 1013.4; 0, 2921.1, 757.3; 0, 0, 1]`）
- 底部标注：`Reprojection error: 0.076 px`

### Stage 2: 目标初始化
- 左侧：一张电机轴示意图（圆柱体端面 + 箭头标注 "shaft φ=12mm"）
- 中间：虚线方框（代表 ROI）包裹住轴的圆形截面
- 右侧输出：`(c_x, c_y, a, b, φ)₀` 文字

### Stage 3: STARK 跟踪
- 左侧：帧序列示意（3 个小矩形排成时间线：t-1, t, t+1）
- 中间：标有 "Transformer Encoder + Decoder" 的矩形框
- 右侧输出：`bbox_t = [x, y, w, h]`
- 可选：一个小的注意力热力图示意（2×2 grid 带颜色深浅）

### Stage 4: 椭圆精检测
- 4 个小方块横向排列，每个内含操作名
- CLAHE 块 → Canny 块 → Hough 块 → fitEllipse 块
- 箭头上方标注中间产物："Enhanced" / "Edges" / "Circle seed" / "Ellipse"
- 右侧输出：`(u_t, v_t, a_t, b_t)`
- 可选：在 fitEllipse 块旁画一个小椭圆示意

### Stage 5: 比例尺 + 振动分析
- 左侧公式框（红框强调）：`s = D·a / (2b²)` 和 `θ = arccos(b/a)`
- 向下箭头 → 换算框：`ΔX = s·Δu, ΔY = s·Δv`
- 向下箭头 → 振动结果框：`A_X = max(X̃) - min(X̃)`
- 可选：右侧放一张小的振动波形示意（横轴 frame，纵轴 displacement μm）

---

## TikZ 生成代码模板（推荐）

```latex
% fig1_pipeline.tex — 独立编译，输出 PDF 后 \includegraphics 到论文
\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}
\usepackage{amsmath}

\begin{document}
\begin{tikzpicture}[
    node distance=8mm,
    box/.style={rectangle, draw=#1, fill=#2, rounded corners=3pt, 
                minimum width=2.2cm, minimum height=1cm, align=center,
                font=\footnotesize},
    box/.default={blue!40}{blue!5},
    arrow/.style={-{Stealth}, thick, gray},
]

% === Stage 1 ===
\node[box={blue!60}{blue!8}, minimum width=14cm] (s1) at (0,0) {
    \textbf{Stage 1: Camera Calibration (Zhang's method)}
};
% ... 继续画 5 个 stage

\end{tikzpicture}
\end{document}
```

---

## 替代方案：Python matplotlib 快速生成

如果不方便用 TikZ，可以用以下 Python 脚本在 `paper/figures/` 下运行生成 PNG：

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# === Stage boxes ===
stages = [
    (0.5, 8.0, "1. Camera Calibration\nCheckerboard → K matrix"),
    (0.5, 6.2, "2. Target Initialization\nSelect ROI → Ellipse fit"),
    (0.5, 4.4, "3. STARK Tracking\nFrame t → bbox [x,y,w,h]"),
    (0.5, 2.6, "4. Ellipse Refinement\nCLAHE → Canny → Hough → fitEllipse"),
    (0.5, 0.8, "5. Scale & Vibration\ns = D·a/(2b²) → X = s·Δu"),
]

for (x, y, label) in stages:
    rect = FancyBboxPatch((x, y), 11.0, 1.4,
                          boxstyle="round,pad=0.1",
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 5.5, y + 0.7, label, ha='center', va='center', fontsize=8,
            fontfamily='monospace')

# === Arrows between stages ===
for i in range(4):
    ax.annotate('', xy=(6, 8.0 - i*1.8 + 0.1), xytext=(6, 8.0 - i*1.8 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

plt.tight_layout()
plt.savefig('fig1_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```
