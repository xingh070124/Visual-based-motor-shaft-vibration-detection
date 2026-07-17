"""Generate Fig. 2: Perspective Projection Geometry Diagram.

Three-panel layout:
  (a) 3D side view — camera + tilted circular cross-section
  (b) Circle-to-ellipse projection relationship
  (c) Image plane detail with ellipse axes and core formula

Output: paper/figures/fig2_geometry.png (250 DPI, white background)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Arc
import numpy as np
import os

FIG_DIR = os.path.dirname(os.path.abspath(__file__))

fig = plt.figure(figsize=(14, 5.5), facecolor="white")

# ============================================================
# Panel (a): 3D Side View -- Camera + Tilted Circle
# ============================================================
ax1 = fig.add_subplot(131)
ax1.set_xlim(-0.5, 8)
ax1.set_ylim(-2, 3)
ax1.set_aspect("equal")
ax1.axis("off")

# Camera body
camera = FancyBboxPatch(
    (-0.3, -0.8), 1.0, 1.6,
    boxstyle="round,pad=0.1",
    facecolor="#ECEFF1", edgecolor="#546E7A", linewidth=2,
)
ax1.add_patch(camera)
ax1.text(0.2, 0.0, "Camera", ha="center", va="center",
         fontsize=9, fontweight="bold", color="#37474F")

# Lens
lens = plt.Circle((0.7, 0), 0.25, facecolor="#90CAF9",
                  edgecolor="#1565C0", linewidth=1.5)
ax1.add_patch(lens)

# Optical axis (dashed)
ax1.plot([0.95, 7.0], [0, 0], "k--", linewidth=1, alpha=0.4)
ax1.annotate("Optical axis", xy=(7.0, 0), xytext=(6.8, 0.3),
             fontsize=7, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# Tilted circle parameters
Z0_x = 5.5
R = 1.2
theta_deg = 20
theta = np.radians(theta_deg)

# Draw tilted circle as ellipse in side view (foreshortened vertically)
tilt_ellipse = Ellipse(
    (Z0_x, 0), R * 2, R * 2 * np.cos(theta),
    facecolor="none", edgecolor="#FF7043", linewidth=2.5,
)
ax1.add_patch(tilt_ellipse)

# Center point
ax1.plot(Z0_x, 0, "o", color="#C62828", markersize=6, zorder=5)
ax1.text(Z0_x + 0.2, -0.3, r"$(0,0,Z_0)$", fontsize=8, color="#C62828")

# Normal vector n
n_len = 1.8
ax1.annotate(
    "", xy=(Z0_x + n_len * np.sin(theta), n_len * np.cos(theta)),
    xytext=(Z0_x, 0),
    arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=2),
)
ax1.text(
    Z0_x + n_len * np.sin(theta) + 0.1,
    n_len * np.cos(theta) + 0.1,
    r"$\mathbf{n}=(\sin\theta,0,\cos\theta)^{\top}$",
    fontsize=7, color="#7B1FA2",
)

# Tilt angle arc
arc_radius = 1.0
arc_angles_deg = np.linspace(90, 90 - theta_deg, 30)
arc_x = Z0_x + arc_radius * np.cos(np.radians(arc_angles_deg))
arc_y = arc_radius * np.sin(np.radians(arc_angles_deg)) - 0.3
ax1.plot(arc_x, arc_y, color="#F9A825", linewidth=2)
ax1.text(Z0_x + 0.5, 0.8, r"$\theta$", fontsize=12,
         color="#F9A825", fontweight="bold")

# Z0 distance annotation (double-headed arrow)
ax1.annotate("", xy=(Z0_x, -1.5), xytext=(0.7, -1.5),
             arrowprops=dict(arrowstyle="<->", color="#546E7A", lw=1.5))
ax1.text(Z0_x / 2 + 0.3, -1.8, r"$Z_0$", fontsize=10,
         ha="center", color="#546E7A")

# Radius label
ax1.annotate(r"$R=D/2$", xy=(Z0_x + 0.8, 0.6), fontsize=8, color="#FF7043",
             arrowprops=dict(arrowstyle="->", color="#FF7043", lw=1))

# Image plane (vertical line)
ax1.plot([-0.2, -0.2], [-2, 2], color="#90A4AE", linewidth=2)
ax1.text(-0.2, -2.3, "Image\nplane", ha="center", fontsize=7, color="#90A4AE")

ax1.set_title("(a) 3D Side View", fontsize=11, fontweight="bold", pad=10)

# ============================================================
# Panel (b): Frontal projection -- circle to ellipse
# ============================================================
ax2 = fig.add_subplot(132)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-3.5, 3.5)
ax2.set_aspect("equal")
ax2.axis("off")

# Left: 3D tilted circle (isometric-ish view)
circle_3d = Ellipse(
    (-2.5, 1.5), 2.0, 1.7, angle=15,
    facecolor="#FFF3E0", edgecolor="#FF7043", linewidth=2,
)
ax2.add_patch(circle_3d)
ax2.text(-2.5, 2.6, "Tilted circle\n(radius R, tilt " + r"$\theta$" + ")",
         ha="center", fontsize=8, color="#E65100")

# Right: projected ellipse on image plane
a_px, b_px = 1.8, 1.3
proj_ellipse = Ellipse(
    (2.0, 1.5), a_px * 2, b_px * 2,
    facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2.5,
)
ax2.add_patch(proj_ellipse)
ax2.text(2.0, 2.6, "Image ellipse\n(semi-axes a, b)",
         ha="center", fontsize=8, color="#0D47A1")

# Projection arrow between them
ax2.annotate("", xy=(0.3, 1.5), xytext=(-0.3, 1.5),
             arrowprops=dict(arrowstyle="->", color="#424242", lw=2))
ax2.text(0.0, 1.2, "Perspective\nprojection",
         ha="center", fontsize=7, color="#424242")

# Key equation box
eq_box = FancyBboxPatch(
    (-3.5, -2.2), 7.0, 1.3,
    boxstyle="round,pad=0.15",
    facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5,
)
ax2.add_patch(eq_box)
ax2.text(0, -1.55, r"$\mathbf{\frac{b}{a} = \cos\theta}$",
         ha="center", va="center", fontsize=14,
         fontweight="bold", color="#C62828")
ax2.text(
    0, -2.7,
    r"$a = \dfrac{F\cdot R}{Z_0\cos^2\theta}\qquad "
    r"b = \dfrac{F\cdot R}{Z_0\cos\theta}$",
    ha="center", va="center", fontsize=10, color="#424242",
)

ax2.set_title("(b) Circle-to-Ellipse Projection",
              fontsize=11, fontweight="bold", pad=10)

# ============================================================
# Panel (c): Image plane detail + final derivation
# ============================================================
ax3 = fig.add_subplot(133)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3.5, 3.5)
ax3.set_aspect("equal")
ax3.axis("off")

# Large ellipse on image plane
a, b = 2.2, 1.6
img_ellipse = Ellipse(
    (0, 1.0), a * 2, b * 2,
    facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2.5,
)
ax3.add_patch(img_ellipse)

# Major axis arrow (a) -- green
ax3.annotate("", xy=(a, 1.0), xytext=(0, 1.0),
             arrowprops=dict(arrowstyle="<->", color="#2E7D32", lw=2.5))
ax3.text(a / 2, 0.6, r"$a$", fontsize=13, ha="center",
         color="#2E7D32", fontweight="bold")

# Minor axis arrow (b) -- blue
ax3.annotate("", xy=(0, 1.0 + b), xytext=(0, 1.0),
             arrowprops=dict(arrowstyle="<->", color="#1565C0", lw=2.5))
ax3.text(-0.4, 1.0 + b / 2, r"$b$", fontsize=13, va="center",
         color="#1565C0", fontweight="bold")

# Center point
ax3.plot(0, 1.0, "o", color="#C62828", markersize=7, zorder=5)
ax3.text(0.2, 0.7, r"$(c_x,c_y)$", fontsize=9, color="#C62828")

# Orientation angle phi arc
phi_arc = Arc(
    (0, 1.0), a * 0.8, b * 0.8, angle=0,
    theta1=0, theta2=18, color="#F9A825", linewidth=2,
)
ax3.add_patch(phi_arc)
ax3.text(0.55, 1.25, r"$\phi$", fontsize=9, color="#F9A825")

# Faint bounding box for scale context
bbox = Rectangle(
    (-a, 1.0 - b), a * 2, b * 2,
    fill=False, linestyle=":", linewidth=1, edgecolor="#BDBDBD",
)
ax3.add_patch(bbox)

# Step-by-step derivation cues (small text)
steps = [
    r"$\mathbf{1.}\;\; \cos\theta = b/a$   (tilt from axes)",
    r"$\mathbf{2.}\;\; \frac{Z_0}{F} = \frac{R}{b\cos\theta} = \frac{Ra}{b^2}$",
    r"$\mathbf{3.}\;\; s = \dfrac{Z_0}{F} = \dfrac{D\cdot a}{2b^2}$",
]
for i, st in enumerate(steps):
    ax3.text(-2.0, -1.5 - i * 0.35, st, fontsize=6.5, color="#616161",
             fontfamily="monospace")

# Core formula box (red, prominent, at bottom)
final_box = FancyBboxPatch(
    (-2.8, -3.35), 5.6, 1.9,
    boxstyle="round,pad=0.2",
    facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2.5,
)
ax3.add_patch(final_box)
ax3.text(0, -2.2,
         r"$\mathbf{s = \dfrac{D \cdot a}{2\,b^{2}}}$",
         ha="center", va="center", fontsize=17,
         fontweight="bold", color="#C62828")
ax3.text(0, -2.85,
         r"$\theta = \arccos\left(\dfrac{b}{a}\right)$",
         ha="center", va="center", fontsize=13, color="#424242")
ax3.text(0, -3.55,
         "Perspective-Corrected Isotropic Scale Factor",
         ha="center", va="center", fontsize=7,
         style="italic", color="#757575")

ax3.set_title("(c) Image Plane Detail & Core Formula",
              fontsize=11, fontweight="bold", pad=10)

# ---- Save ----
plt.tight_layout(pad=1.5)
outpath = os.path.join(FIG_DIR, "fig2_geometry.png")
plt.savefig(outpath, dpi=250, bbox_inches="tight", facecolor="white")
plt.close()

size_kb = os.path.getsize(outpath) / 1024
print(f"Fig. 2 saved: {outpath}")
print(f"Size: {size_kb:.1f} KB")
print("Done.")
