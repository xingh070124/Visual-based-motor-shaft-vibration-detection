# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 文件结构

src/    # main原始文件
build/  # 构建文件（含原始脚本备份）
docs/   # 文档
test/   # 测试文件(包含测试视频，图片等)
|—— figures/
|—— video/
utils/  # 工具文件(包含工具函数等)

## 项目概述

基于视觉的电机轴振动检测系统——本科毕业论文项目（李昕泽，学号 221001700218）。

核心思路：使用 STARK 视觉目标跟踪算法追踪高速旋转电机轴的像素运动轨迹，通过相机内参矩阵将像素坐标转换为真实相机坐标系下的物理位移，最终计算电机轴的振动幅度。同时辅以激光位移传感器数据进行对比/融合分析。

## 运行环境

项目固定运行在 **base 环境 `D:\anaconda\python.exe`**（已含 torch+CUDA、opencv-contrib、openpyxl、tensorboardX、pandas 全部依赖，STARK 完整可跑）。

> ⚠️ 不再使用 conda 虚拟环境 `stark`。`stark` 环境目前缺 torch，无法跑完整 STARK。如需改回 `stark`，须自行安装 PyTorch+CUDA 后再切换。

- Python 3.x
- OpenCV (`opencv-contrib-python`，含 `TrackerCSRT_create`)、NumPy、Pandas、openpyxl
- PyTorch + CUDA（STARK 跟踪器依赖）
- **STARK 代码库**（外部依赖）：项目代码通过 `src/tracking/stark_wrapper.py` 自动将 STARK 根目录注入 `sys.path`，并自动生成所需的 `local.py` 配置文件
- **STARK 预训练权重**：需从 STARK MODEL_ZOO 下载 `STARKST_ep0050.pth.tar`，放到 `D:\github\clone\Stark\checkpoints\train\stark_st2\baseline\` 目录下

通过项目根目录的 `run.bat` 启动，**无需 `conda activate`**（绝对路径调用 base 解释器，绕开 conda 激活与 PATH 混乱，避免 `ModuleNotFoundError`）。VS Code 已通过 `.vscode/settings.json` 固定解释器为 `D:\anaconda\python.exe`。

## 运行方式

```bash
# 推荐：用启动器（固定 base 环境，无需 conda activate）
run.bat                 # 批量视频分析（默认入口，src.batch_analysis）
run.bat main            # 图像序列主流程（src.main）
run.bat --skip-render   # 透传任意参数给 batch_analysis

# 也可直接用绝对路径解释器
D:\anaconda\python.exe -m src.main
D:\anaconda\python.exe -m src.batch_analysis

# 相机标定
D:\anaconda\python.exe -m src.calibration.calibrator

# 激光数据处理
D:\anaconda\python.exe -m src.processing.laser_data
```

## 源码模块说明

```
src/
├── config.py                     # 集中配置（所有硬编码参数统一管理）
├── calibration/
│   ├── calibrator.py             # 张氏标定法（CameraCalibrator 类）
│   └── scale_factor.py           # 逐图像尺度因子计算
├── tracking/
│   ├── stark_wrapper.py          # STARK-ST 跟踪器封装（自动注入 sys.path + 生成 local.py）
│   ├── shaft_detector.py         # ROI内电机轴圆形轮廓检测
│   └── coordinate.py             # 像素坐标 → 相机坐标转换
├── processing/
│   └── laser_data.py             # 激光位移传感器数据后处理
├── vibration/
│   └── analyzer.py               # 振动幅度计算与分析
└── main.py                       # 主流程入口
utils/
└── file_utils.py                 # 文件工具（自然排序、图像查找）
build/                            # 原始脚本备份（5个脚本，内容不变）
├── 相机/  (cameraCalibration.py, world s.py)
├── 激光/  (yuanqiujie.py, V ...py)
└── STARK/ (stark_tracking_camera_coords.py)
```

## 核心流程（src/main.py）

1. **加载图像序列**：从 `frames200-1/` 读取 PNG 帧，自然排序
2. **手动选 ROI**：首帧交互式框选目标区域（`cv2.selectROI`）
3. **轮廓精确检测**：在 ROI 内 Canny 边缘检测 → 轮廓圆度筛选 → 最小包围圆 → 半径校验
4. **STARK-ST 跟踪**：自动注入 STARK 路径、生成 local.py，加载预训练权重初始化跟踪器
5. **坐标转换**：`(u, v) → (X, Y) = Z * K⁻¹ @ [u, v, 1]ᵀ`
6. **振动计算**：`amplitude = max - min`（X/Y 方向 + 欧氏总幅度）
7. **输出**：Excel 文件（帧号/像素坐标/相机坐标/振幅）

## 关键参数

所有参数集中在 `src/config.py`，修改该文件即可调整：

- **相机内参 (一倍)**：`fx=871.63, fy=1159.30, cx=375.11, cy=210.87`
- **深度 Z**：0.1817 m
- **电机轴直径**：12 mm
- **STARK 根目录**：`D:\github\clone\Stark`
- **预训练权重**：`checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar`

## 数据文件说明

- 标定图像：`./images001/`, `./images002/` 目录下的 JPG 文件
- 跟踪帧：`./frames200-1/` 目录下的 PNG 序列
- 激光原始数据：`outputleft down.xlsx`, `9righ down.xlsx` 等 Excel 文件
- 跟踪结果输出：`stark_tracking_camera_coords.xlsx`

## 注意事项

- 深度 Z 为预设固定值，未使用 PnP 或深度估计实时计算
- STARK 跟踪器首次运行时需 GPU (CUDA)，`stark_wrapper.py` 会在导入时自动配置环境
- 激光数据处理脚本内硬编码了偏移量/缩放因子，可通过 `process_laser_data()` 参数灵活调用
