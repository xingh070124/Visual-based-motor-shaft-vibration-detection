# docs/ — 项目文档与参考文献

## 研究文档

| 文件 | 内容 |
|------|------|
| [literature_review.md](literature_review.md) | 文献综述：针对圆形检测不稳定性问题的 10 篇论文分类综述 |
| [architecture_evaluation.md](architecture_evaluation.md) | 架构升级可行性评估：CNN 检测、学习特征 Hough、SeqTrack 替代方案 |
| [tilt_correction_research_plan.md](tilt_correction_research_plan.md) | 电机轴倾斜场景下振动测量误差补偿：数学推导与实验回路设计 |
| [tilt_solutions_analysis.md](tilt_solutions_analysis.md) | 非垂直安装解决方案分析：纯软件补偿 / 多传感器融合 / 硬件对准 |

## 参考论文 (papers/)

### circle_detection/ — 圆形检测 (3篇)

| 编号 | arXiv | 标题 | 关键贡献 |
|------|-------|------|---------|
| A | 1204.0382 | Feature Detection in Radio Astronomy using CHT | CHT 噪声响应分析，置信度过滤假阳性 |
| B | 2003.04676 | Deep Hough Transform for Semantic Line Detection | 深度学习特征 + Hough 投票融合 |
| C | 2301.09657 | YOLO-CL: Galaxy cluster detection | YOLO 改造为圆形目标检测 |

### visual_tracking/ — 视觉跟踪与运动测量 (3篇)

| 编号 | arXiv | 标题 | 关键贡献 |
|------|-------|------|---------|
| D | 2311.03725 | DeepInspect: AI-Powered Defect Detection | CNN+RNN 工业缺陷检测 pipeline |
| J | 2304.14394 | Unified Seq2Seq Learning for Visual Object Tracking | Transformer seq2seq 跟踪（SeqTrack） |
| - | 2101.07005 | Optical Flow Method for Measuring Deformation | 光流法测量土样剪切变形 |

### camera_calibration/ — 相机标定与位姿估计 (4篇)

| 编号 | arXiv | 标题 | 关键贡献 |
|------|-------|------|---------|
| I | 2303.10559 | Deep Learning for Camera Calibration: A Survey | 深度学习相机标定方法综述 |
| - | 1907.10219 | Efficient Circle-Based Camera Pose Tracking Free of PnP | 圆形标记无 PnP 位姿跟踪 |
| - | 2212.03239 | Perspective Fields for Single Image Camera Calibration | 透视场单图标定 (CVPR 2023) |
| - | 2503.07763 | 2D/3D Registration with Differentiable Ellipse Fitting | 可微椭圆拟合与透视投影 |

### kalman_filtering/ — 卡尔曼滤波 (3篇)

| 编号 | arXiv | 标题 | 关键贡献 |
|------|-------|------|---------|
| E | 2210.12807 | HKF: Hierarchical Kalman Filtering | 分层 KF + 在线学习演化先验 |
| F | 2001.05165 | Terrain estimation via vehicle vibration + CKF | 容积 KF 处理振动数据 |
| G | 2210.04165 | Neural Extended Kalman Filters | 可学习 EKF 分离快慢成分 |

### signal_estimation/ — 信号估计理论 (1篇)

| 编号 | arXiv | 标题 | 关键贡献 |
|------|-------|------|---------|
| H | 1205.6903 | Cramér-Rao Bounds for Polynomial Signal Estimation | AR(1) 漂移下信号估计 CRLB |
