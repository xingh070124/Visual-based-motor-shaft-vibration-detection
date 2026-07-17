# Fig. 2 — Perspective Projection Geometry Diagram 制作提示词

> 目标：IEEE TIM 双栏论文，宽度 ≈ 8.5 cm，展示倾斜圆→椭圆投影的 3D 几何关系
> 推荐工具：TikZ（3D 坐标系 + 投影）> Inkscape > Python matplotlib
> ⚠️ 这是论文最核心的示意图，建议用 TikZ 以获得最佳数学排版

---

## 图片结构：三部分并行展示

```
┌────────────────────┬────────────────────┬─────────────────────┐
│    (a) 3D 侧视图    │   (b) 正视/俯视图   │   (c) 图像平面投影   │
│                    │                    │                     │
│ 相机──→ Z₀ ──→圆面 │  倾斜圆(3D) → 椭圆  │  椭圆半轴标注        │
│ 光轴   /|\ 倾斜角θ  │                    │  a = F·R/(Z₀cos²θ)  │
│       / | \        │                    │  b = F·R/(Z₀cosθ)   │
│      /  |  \       │                    │                     │
└────────────────────┴────────────────────┴─────────────────────┘
```

---

## 方案 A：三面板布局（推荐，信息量最大）

### Panel (a): 3D 侧视图 — 相机 + 倾斜圆面

```
                      Z₀
        相机 ←────────────────────→ 圆面
                                    
         ┌───┐                      ╱
         │   │                     ╱  圆 (半径 R)
         │透 │  ←── 光轴 ──→     ╱   ╭─────╮
         │镜 │                   ╱   ╱       ╲
         │   │                  ╱   │    ●    │  ← 圆心在 (0,0,Z₀)
         └───┘                 ╱θ   ╲       ╱
        图像平面               ╱      ╰─────╯
        (像方)                      法线 n = (sinθ, 0, cosθ)
```

**关键标注：**
- 相机（左侧小矩形 + 透镜符号）
- 光轴虚线（水平）
- 距离标注 `Z₀`（相机→圆面中心）
- 倾斜圆（椭圆形状示意，倾斜角 θ 用弧线 + 标签）
- 法线向量 `n = (sinθ, 0, cosθ)ᵀ`
- 圆半径标注 `R = D/2 = 6 mm`
- 图像平面（相机左侧竖线）

### Panel (b): 正视投影 — 圆→椭圆

```
        3D 空间                        图像平面投影
    ┌──────────┐                   ┌──────────────┐
    │  ╭─────╮ │                   │    ╭─────╮    │
    │ ╱   ●   ╲│     透视投影       │  ╱   ●   ╲   │
    │ │   R   ││  ──────────────→  │  │       │   │
    │ ╲       ╱│                   │  ╲       ╱   │
    │  ╰─────╯ │                   │   ╰─────╯    │
    │ 倾斜圆   │                   │   椭圆(a,b)  │
    └──────────┘                   └──────────────┘
         ↑                               ↑
    半径 R, 倾角 θ               a > b, b/a = cosθ
```

**关键标注：**
- 左侧：正圆形，标注 `半径 R`，倾角 `θ` 用斜线表示不是正视
- 箭头 → 标注 "perspective projection"（或 "透视投影"）
- 右侧：椭圆，半长轴 `a`（水平方向）和半短轴 `b`（竖直方向）用带箭头的线段标出
- 公式标注：`b/a = cosθ`，`a = F·R/(Z₀·cos²θ)`，`b = F·R/(Z₀·cosθ)`

### Panel (c): 图像平面上的椭圆放大

```
                  ┌──────────────────────┐
                  │                      │
       a (长轴) → │    ╭──────────╮      │
                  │   ╱            ╲     │
                  │  │      ●       │    │ ← 中心 (cₓ,c_y)
                  │   ╲            ╱     │
                  │    ╰──────────╯      │
                  │     ← b (短轴) →      │
                  │                      │
                  └──────────────────────┘
                        图像平面 (像素坐标)
                        
        scale = D·a / (2b²)     θ = arccos(b/a)
```

**关键标注：**
- 椭圆外接矩形（虚线）
- 半长轴 `a`：用箭头从中心→椭圆边界，标注 `a`
- 半短轴 `b`：用箭头从中心→椭圆边界（垂直于 a），标注 `b`
- 中心点 `(cₓ, c_y)` 用小圆点标记
- 长轴方向角 `φ`（可选弧线）
- 下方公式框（红框强调）：`s = D·a/(2b²)`，`θ = arccos(b/a)`

---

## 方案 B：单图紧凑布局（备选，适合空间紧张时）

如果双栏宽度不够放三个面板，用一个 3D 等轴测图同时展示所有信息：

```
                        图像平面
                        ┌──────┐
                        │ ╭──╮ │ ← 椭圆投影 (a,b)
                        │ │● │ │
                        │ ╰──╯ │
                        └──────┘
                          ↑
                          │ 投影光线
                          │
            ╱─────╲
           ╱   ●   ╲  ← 倾斜圆 (半径 R, 倾角 θ)
           ╲       ╱
            ╰─────╯
              ↑
              │ Z₀
          ┌───┴───┐
          │ 相机   │
          │  O    │
          └───────┘
```

---

## 配色方案

| 元素 | 颜色 (HEX) | 说明 |
|------|-----------|------|
| 相机/透镜 | `#546E7A` | 蓝灰色填充 |
| 光轴 | `#90A4AE` (虚线) | 浅灰虚线 |
| 3D 圆（空间） | `#FF7043` (边框) | 橙色描边 |
| 椭圆投影（像面） | `#42A5F5` (填充 + 边框) | 蓝色 |
| 中心点 | `#C62828` | 深红圆点 d=4pt |
| a 标注线 | `#2E7D32` | 绿色箭头 |
| b 标注线 | `#1565C0` | 深蓝箭头 |
| 公式框 | `#C62828` (边框) / `#FFEBEE` (填充) | 红框浅红底 |
| 角度弧线 | `#F9A825` | 黄色弧线 |
| 文字标注 | `#212121` | 深灰/黑色 |

---

## TikZ 代码模板

```latex
% fig2_geometry.tex — 独立编译
\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{3d, perspective, arrows.meta, angles, quotes, 
                calc, decorations.pathreplacing, shapes.geometric}
\usepackage{amsmath}

\begin{document}
\begin{tikzpicture}[
    scale=1.2,
    % 3D coordinate system
    x={(1cm,0.3cm)}, y={(0cm,1cm)}, z={(-0.8cm,-0.3cm)},
    % Styles
    camera/.style={draw=blue!50, fill=blue!5, minimum width=1.2cm, 
                   minimum height=0.8cm},
    ellipse_proj/.style={draw=blue!60, fill=blue!10, dashed},
    dim_arrow/.style={-{Stealth[length=3mm]}, thick},
]

% === Camera at origin ===
\node[camera] (cam) at (0,0,0) {Camera};
\draw[->, thick] (0,0,0) -- (0,0,5) node[above] {$Z$ (optical axis)};

% === Tilted circle at Z=Z₀ ===
\def\R{1.2}   % circle radius
\def\Zzero{4} % Z₀ distance
\def\Tilt{20} % tilt angle in degrees

% Draw tilted circle (approximate with ellipse in 3D)
\draw[orange, thick] 
    plot[domain=0:360, samples=60, variable=\p]
    ({\R*cos(\p)}, {\R*cos(\Tilt)*sin(\p)}, {\Zzero - \R*sin(\Tilt)*sin(\p)});

% Center point of circle
\fill[red] (0,0,\Zzero) circle (2pt) node[above right] {$(0,0,Z_0)$};

% Normal vector
\draw[->, thick, purple] (0,0,\Zzero) -- 
    ({sin(\Tilt)*1.5}, 0, {\Zzero + cos(\Tilt)*1.5})
    node[right] {$\mathbf{n} = (\sin\theta, 0, \cos\theta)^\top$};

% Tilt angle arc
\draw[->, thick, yellow!80!orange] (0,0,\Zzero-1.5) 
    arc[start angle=90, end angle={90-\Tilt}, radius=1.5]
    node[midway, left] {$\theta$};

% === Image plane projection ===
% Simplified: show ellipse on image plane
\begin{scope}[shift={(-3cm, 0)}]
    % Image plane
    \draw[gray, thick] (0,-2) -- (0,3);
    \node[gray] at (0.3,3.2) {image plane};
    
    % Projected ellipse
    \def\a{1.3} % semi-major axis
    \def\b{0.95} % semi-minor axis
    
    \draw[blue!60, fill=blue!8, thick]
        (0,0) ellipse ({\a} and {\b});
    
    % Mark axes
    \draw[dim_arrow, green!60!black] (0,0) -- ({\a},0) 
        node[midway, below] {$a$};
    \draw[dim_arrow, blue!60] (0,0) -- (0,{\b}) 
        node[midway, left] {$b$};
    
    % Center
    \fill[red] (0,0) circle (1.5pt) node[below right] {$(c_x,c_y)$};
    
    % Formula box at bottom
    \node[draw=red!60, fill=red!5, rounded corners, 
          text width=3.5cm, align=center, font=\footnotesize] 
          at (0,-3.5) {
        $s = \dfrac{D \cdot a}{2b^{2}}$ \\[4pt]
        $\theta = \arccos\!\left(\dfrac{b}{a}\right)$
    };
\end{scope}

\end{tikzpicture}
\end{document}
```

---

## 替代方案：Python matplotlib 3D

如果不想用 TikZ，用以下 Python 脚本（需要 `pip install matplotlib numpy`）：

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, FancyBboxPatch

fig = plt.figure(figsize=(10, 4))

# === Subplot 1: 3D side view ===
ax1 = fig.add_subplot(131, projection='3d')
R, Z0, theta = 6, 250, np.radians(15)

# Camera
ax1.scatter([0], [0], [0], c='blue', s=100, marker='s', label='Camera')
ax1.plot([0, 0], [0, 0], [0, Z0], 'k--', alpha=0.5, label='Optical axis')

# Circle points
phi = np.linspace(0, 2*np.pi, 100)
Xc = R * np.cos(phi)
Yc = R * np.cos(theta) * np.sin(phi)
Zc = Z0 - R * np.sin(theta) * np.sin(phi)
ax1.plot(Xc, Yc, Zc, 'orange', linewidth=2, label='Shaft circle (tilted)')
ax1.scatter([0], [0], [Z0], c='red', s=30, label='Center (0,0,Z₀)')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('(a) 3D Side View')
ax1.legend(fontsize=6)

# === Subplot 2: Image plane projection ===
ax2 = fig.add_subplot(132)
ax2.set_aspect('equal')
# Projected ellipse: a ≈ F*R/(Z0*cos²θ), b ≈ F*R/(Z0*cosθ)
F = 2928.5
a_px = F * R / (Z0 * np.cos(theta)**2) / 100  # scale down for display
b_px = F * R / (Z0 * np.cos(theta)) / 100

ellipse = Ellipse((0, 0), 2*a_px, 2*b_px, angle=0,
                  facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax2.add_patch(ellipse)
ax2.plot(0, 0, 'ro', markersize=4)
ax2.annotate('', xy=(a_px, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(a_px/2, -0.3, 'a', color='green', ha='center')
ax2.annotate('', xy=(0, b_px), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.text(-0.3, b_px/2, 'b', color='blue', va='center')
ax2.set_xlim(-a_px*1.8, a_px*1.8)
ax2.set_ylim(-b_px*1.8, b_px*1.8)
ax2.set_title('(b) Image Plane Projection')
ax2.axis('off')
ax2.text(0, -b_px*1.6, r'$b/a = \cos\theta$', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828'))

# === Subplot 3: Key equations ===
ax3 = fig.add_subplot(133)
ax3.axis('off')
eq_box = FancyBboxPatch((0.1, 0.3), 0.8, 0.4,
                        boxstyle="round,pad=0.15",
                        facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
ax3.add_patch(eq_box)
ax3.text(0.5, 0.6, r'$\mathbf{s = \frac{D \cdot a}{2b^{2}}}$',
         ha='center', va='center', fontsize=13, fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0.5, 0.4, r'$\theta = \arccos\left(\frac{b}{a}\right)$',
         ha='center', va='center', fontsize=12,
         transform=ax3.transAxes)
ax3.text(0.5, 0.2, 'Perspective-Corrected\nIsotropic Scale',
         ha='center', va='center', fontsize=8, style='italic',
         transform=ax3.transAxes)
ax3.set_title('(c) Core Formula')

plt.tight_layout()
plt.savefig('fig2_geometry.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

---

## 快速检查清单

- [ ] 相机在左，光轴水平向右
- [ ] 圆形截面在 Z₀ 处，倾斜角 θ 用弧线标注
- [ ] 法线向量方向标注
- [ ] 像平面上的椭圆，a（长轴）和 b（短轴）明确标注
- [ ] 公式 b/a = cosθ 和 s = D·a/(2b²) 在图中呈现
- [ ] 配色与论文整体协调（蓝/橙/红/绿四色系）
- [ ] 所有文字 ≥ 8pt（IEEE 要求）
- [ ] 箭头样式统一，线宽 ≥ 1pt
- [ ] 保存为 PDF（矢量）用于 LaTeX，或 300 DPI PNG 作为备选
