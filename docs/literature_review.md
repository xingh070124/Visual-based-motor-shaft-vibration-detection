# 电机轴圆形检测系统不稳定性问题 — 文献综述

> **背景**：基于视频的电机轴圆形特征检测系统存在严重不稳定性——自适应半径估计导致正反馈崩溃、比例尺膨胀、振动信号抖动等问题。本综述针对诊断报告中的8个根因，按问题域分类综述了10篇相关论文。

---

## 1. 自适应参数估计与正反馈问题（问题#1、#3、#4）

### 1.1 问题描述

检测系统使用Hough圆变换的自适应半径更新机制：`expected_radius = vis_radius`（前一帧检测结果直接赋值给下一帧）。这导致正反馈循环——半径估计偏差在帧间累积，最终导致检测完全崩溃（从342px缩至75px，或膨胀至1070px）。同时，半径搜索范围[0.4, 1.8]倍过宽，导致轴承外圈被误检为shaft圆，产生双峰直方图和scale跳变。

### 1.2 相关论文

**论文A：Feature Detection in Radio Astronomy using the Circle Hough Transform**
- Hollitt & Johnston-Hollitt, 2012, arXiv:1204.0382
- 发表于Publications of the Astronomical Society of Australia
- **核心贡献**：系统分析了CHT在噪声条件下的响应特性，区分了"纯噪声产生的参数空间峰值"与"真实圆形目标产生的峰值"。提出基于置信度统计量的假阳性过滤方法。
- **与本问题的关系**：当前系统的半径搜索范围[0.4, 1.8]倍过宽，导致噪声边缘和轴承外圈都能在Hough空间产生峰值。论文提出的置信度阈值方法可以直接应用于过滤这些假阳性检测——不是缩小搜索范围（会丢失真实目标），而是提高检测的统计置信度要求。

**论文B：Deep Hough Transform for Semantic Line Detection**
- Zhao, Han, Zhang, Xu & Cheng, 2021, arXiv:2003.04676
- IEEE TPAMI 2021 (DOI: 10.1109/TPAMI.2021.3077129)
- **核心贡献**：将深度学习特征提取与Hough变换融合，在参数空间中引入可学习的注意力权重。用CNN提取的语义特征替代传统梯度投票，使得检测对噪声和遮挡更鲁棒。
- **与本问题的关系**：这是从根本上解决正反馈问题的方案——不再依赖前一帧的半径估计值，而是让网络从图像特征中直接推断圆参数。即使作为折中方案，也可以借鉴其"在Hough空间加注意力权重"的思路，对每个投票的贡献度加权，抑制噪声驱动的异常投票。

**论文C：YOLO-CL: Galaxy cluster detection with deep machine learning**
- Grishin, Mei & Ilic, 2023, arXiv:2301.09657
- Astronomy & Astrophysics (DOI: 10.1051/0004-6361/202345976)
- **核心贡献**：将YOLO目标检测架构改造为检测延展性圆形目标（星系团），实现95-98%检测率。用端到端学习替代参数化Hough变换。
- **与本问题的关系**：这是最彻底的升级方案。YOLO类检测器天然不存在"自适应参数正反馈"问题，因为每帧独立检测，不依赖前一帧状态。虽然该论文面向天文领域，但其"将YOLO改造为圆形目标检测"的方法论可以直接迁移到电机轴检测场景。

**论文D：DeepInspect: An AI-Powered Defect Detection for Manufacturing Industries**
- Kumbhar et al., 2023, arXiv:2311.03725
- **核心贡献**：将CNN+RNN+GAN组合用于制造业缺陷检测，提供完整的工业视觉检测pipeline架构。用CNN做单帧检测、RNN做时序一致性验证。
- **与本问题的关系**：其多模型融合架构可以借鉴——用CNN检测shaft圆+RNN验证帧间一致性，既解决单帧误检也解决帧间不一致。

### 1.3 小结

自适应Hough参数的正反馈问题有三条解决路径：
1. **短期修复**（论文A）：增加统计置信度阈值，过滤假阳性投票
2. **中期改进**（论文B）：用学习特征替代梯度投票，减少对参数自适应的依赖
3. **长期升级**（论文C/D）：用深度学习检测器完全替代Hough变换

---

## 2. 振动信号时域滤波与Kalman滤波（问题#5）

### 2.1 问题描述

STARK跟踪器的bbox中心坐标(y_center)直接作为振动信号使用，无任何时域滤波。导致高频亚像素抖动（~1px随机波动）叠加在真实振动上，降低振动分析的信噪比。10个视频因转速不同，振动频率各异，需要自适应滤波。

### 2.2 相关论文

**论文E：HKF: Hierarchical Kalman Filtering with Online Learned Evolution Priors for Adaptive ECG Denoising**
- Revach, Locher, Shlezinger, van Sloun & Vullings, 2023, arXiv:2210.12807
- IEEE Transactions on Signal Processing
- **核心贡献**：提出分层Kalman滤波框架，通过在线学习信号演化动力学先验，自适应调整状态转移矩阵。比传统固定参数KF更鲁棒，特别适合信号统计特性时变的场景。
- **与本问题的关系**：电机在不同转速下振动频率不同（0.25x~10x），传统固定参数Kalman无法同时适应。HKF的在线学习机制可以自动估计当前转速下的振动频率，动态调整滤波器带宽。论文中ECG信号处理的思路（准周期信号+噪声）与轴振动信号高度相似。

**论文F：Terrain estimation via vehicle vibration measurement and cubature Kalman filtering**
- Reina, Leanza & Messina, 2020, arXiv:2001.05165
- Journal of Vibration and Control (DOI: 10.1177/1077546319890011)
- **核心贡献**：用容积Kalman滤波器（Cubature Kalman Filter, CKF）处理振动传感器数据。证明CKF在非线性系统状态估计中优于标准扩展Kalman滤波器（EKF），在精度和稳定性上均有提升。
- **与本问题的关系**：视频追踪坐标到物理位移的映射是弱非线性的（像素-物理单位转换），CKF比标准KF更适合。论文直接展示了从振动测量数据中估计动态参数的方法，可以迁移到轴振动信号的滤波处理。

### 2.3 小结

对于振动坐标滤波，推荐方案：先用论文F的CKF框架建立基础滤波器，再引入论文E的在线学习机制使滤波器适应不同转速。实现路径：`bbox_center → CKF(x, P, Q, R) → 滤波后坐标`，其中Q和R根据前N帧残差在线更新。

---

## 3. 漂移去除与趋势分离（问题#6）

### 3.1 问题描述

振幅计算使用原始峰峰值(Amplitude = max(v) - min(v))，未去除慢速漂移。热膨胀、轴承磨合等因素导致低频趋势叠加在振动信号上，使得峰峰值虚高，不同视频的振幅基准不一致。

### 3.2 相关论文

**论文G：Neural Extended Kalman Filters for Learning and Predicting Dynamics of Structural Systems**
- Liu, Lai, Bacsa & Chatzi, 2023, arXiv:2210.04165
- Structural Health Monitoring (DOI: 10.1177/14759217231179912)
- **核心贡献**：提出可学习的扩展Kalman滤波器（Neural EKF），将结构动力学状态分为快变（振动）和慢变（漂移/损伤）成分，分别建模和估计。网络自动学习哪些频率成分属于"正常振动"、哪些属于"趋势漂移"。
- **与本问题的关系**：这正是去趋势问题的最优解——不是简单线性拟合去趋势，而是用物理先验指导的Neural EKF自动分离振动（高频，~几十Hz）和热漂移（低频，<0.1Hz）。即使不部署Neural EKF，其"将状态向量分为振动子状态和漂移子状态"的建模思想也可以手动实现。

**论文H：Cramér-Rao Bounds for Polynomial Signal Estimation using Sensors with AR(1) Drift**
- Kar, Varshney & Palaniswami, 2012, arXiv:1205.6903
- IEEE Transactions on Signal Processing (DOI: 10.1109/TSP.2012.2204989)
- **核心贡献**：建立了传感器漂移条件下多项式信号估计的Cramér-Rao下界（CRLB），证明了漂移对估计精度的影响以及最优补偿策略。
- **与本问题的关系**：提供了去趋势的理论最优方案。热膨胀漂移近似线性（论文中多项式漂移模型的一阶特例），可以用其方法建立漂移模型并从振动信号中分离。CRLB还可以帮助评估：在当前漂移水平下，振幅测量的理论精度极限是多少。

### 3.3 小结

去趋势方案分两级：
1. **工程级**（论文H）：用AR(1)模型拟合漂移并减去，简单有效
2. **智能级**（论文G）：用Neural EKF自动分离快慢成分，精度更高但实现复杂

---

## 4. 比例尺校准与像素-物理单位转换（问题#2、#8）

### 4.1 问题描述

scale = physical_radius / R_median，其中R_median为前50帧半径中位数。当检测崩溃导致R异常时，scale被严重污染。且10个视频无统一标定，数据语义不一致。

### 4.2 相关论文

**论文I：Deep Learning for Camera Calibration and Beyond: A Survey**
- Liao, Nie, Huang, Lin, Zhang, Zhao, Gabbouj & Tao, 2023-2025, arXiv:2303.10559
- **核心贡献**：全面综述了基于学习的相机标定方法，包括内参标定、畸变校正、像素-物理单位转换。涵盖传统棋盘格标定到神经网络在线标定的演进。
- **与本问题的关系**：提供了替代脆弱R_median方案的理论基础。论文中"在线标定"方法可以在运行时持续验证scale值的合理性——当检测到scale突变时触发异常检测，而非被动地被错误scale值误导。

### 4.3 小结

比例尺校准的关键是引入冗余校准机制：除了R_median主通道外，增加一个独立的scale验证通道（如已知物理尺寸的参考标记），当两者不一致时触发异常处理。

---

## 5. 跟踪器稳定性（问题#5补充）

### 5.1 相关论文

**论文J：SeqTrack: Sequence to Sequence Learning for Visual Object Tracking**
- Chen et al., 2023, arXiv:2304.14394
- CVPR 2023
- **核心贡献**：将视觉目标跟踪重新表述为seq2seq序列到序列问题，用Transformer编码器-解码器替代复杂的跟踪头（分类+回归分支）。简化了跟踪pipeline，减少了多头预测引入的不一致性。
- **与本问题的关系**：STARK使用分类+回归双头架构，bbox输出可能因两个头的预测不一致而产生亚像素抖动。SeqTrack的统一流式输出理论上更平滑。即使不替换STARK，其bbox回归方式（自回归逐步修正）可以作为后处理参考来平滑坐标。

---

## 6. 综合推荐方案

基于以上文献综述，针对当前系统不稳定性问题，推荐分阶段实施：

### 阶段一：紧急修复（1-2天）
- **论文A方法**：在Hough检测后增加统计置信度过滤，提高minVotes阈值
- **论文H方法**：对振幅信号做AR(1)漂移去除
- **工程修复**：统一高斯模糊预处理、锁定expected_radius在初始值

### 阶段二：信号处理升级（1周）
- **论文E+F方法**：部署CKF+在线学习Kalman滤波器
- **论文I方法**：增加独立scale校准通道

### 阶段三：架构升级（2-4周）
- **论文C/D方法**：用YOLO/CNN检测器替代Hough变换
- **论文B方法**：引入学习特征的Hough变换（折中方案）
- **论文J方法**：评估SeqTrack替代STARK的可行性

---

## 附录：论文索引

| 编号 | arXiv ID | 标题 | 解决问题 |
|------|----------|------|---------|
| A | 1204.0382 | [Feature Detection in Radio Astronomy using CHT](papers/circle_detection/Feature%20Detection%20in%20Radio%20Astronomy%20using%20the%20Circle%20Hough%20Transform.pdf)  | #1,#3,#4 |
| B | 2003.04676 | [Deep Hough Transform for Semantic Line Detection](papers/circle_detection/Deep%20Hough%20Transform%20for%20Semantic%20Line%20Detection.pdf) | #1,#3,#4 |
| C | 2301.09657 | [YOLO-CL: Galaxy cluster detection](papers/circle_detection/YOLO-CL%20Galaxy%20cluster%20detection%20in%20the%20SDSS%20with%20deep%20machine%20learning.pdf) | #1~#4 |
| D | 2311.03725 | [DeepInspect: AI-Powered Defect Detection](papers/visual_tracking/DeepInspect%20An%20AI-Powered%20Defect%20Detection%20for%20Manufacturing%20Industries.pdf) | #1~#4 |
| E | 2210.12807 | [HKF: Hierarchical Kalman Filtering](papers/kalman_filtering/HKF%20Hierarchical%20Kalman%20Filtering%20with%20Online%20Learned%20Evolution%20Priors%20for%20Adaptive%20ECG%20Denoising.pdf) | #5 |
| F | 2001.05165 | [Terrain estimation via vehicle vibration + CKF](papers/kalman_filtering/Terrain%20estimation%20via%20vehicle%20vibration%20measurement%20and%20Cubature%20Kalman%20Filtering.pdf) | #5 |
| G | 2210.04165 | [Neural Extended Kalman Filters](papers/kalman_filtering/Neural%20Extended%20Kalman%20Filters%20for%20Learning%20and%20Predicting%20Dynamics%20of%20Structural%20Systems.pdf) | #6 |
| H | 1205.6903 | [CRLB for Polynomial Signal Estimation with Drift](papers/signal_estimation/Cram%C2%B4er-Rao%20Bounds%20for%20Polynomial%20Signal%20Estimation%20using%20Sensors%20with%20AR(1)%20Drift.pdf) | #6 |
| I | 2303.10559 | [Deep Learning for Camera Calibration Survey](papers/camera_calibration/Deep%20Learning%20for%20Camera%20Calibration%20and%20Beyond%20A%20Survey.pdf) | #2,#8 |
| J | 2304.14394 | [SeqTrack: Seq2Seq Learning for Tracking](papers/visual_tracking/Unified%20Sequence-to-Sequence%20Learning%20for%20Single-%20and%20Multi-Modal%20Visual%20Object%20Tracking.pdf) | #5 |
