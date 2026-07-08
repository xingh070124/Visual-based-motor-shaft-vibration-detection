# -*- coding: utf-8 -*-
"""容积卡尔曼滤波器（Cubature Kalman Filter, CKF）。

用于振动坐标平滑，基于论文F的CKF方法。
CKF使用球面径向容积准则，比EKF更适合非线性坐标转换，
比UKF无需调节alpha/beta/kappa参数。

状态模型: [x, y, vx, vy]（位置+速度）
测量模型: [x, y]（仅位置）
"""

import numpy as np
from typing import Tuple


class CKFFilter:
    """容积卡尔曼滤波器，用于振动坐标平滑。

    状态: [x, y, vx, vy]
    测量: [x, y]
    """

    def __init__(self, dt: float = 1.0 / 71.66, process_noise: float = 1.0,
                 measurement_noise: float = 1.0):
        """初始化CKF。

        Args:
            dt: 时间步长（秒），默认1/71.66（对应71.66fps）
            process_noise: 过程噪声标准差（像素）
            measurement_noise: 测量噪声标准差（像素）
        """
        self.dt = dt
        self.n = 4  # 状态维度
        self.m = 2  # 测量维度

        # 状态转移矩阵（匀速模型）
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # 测量矩阵（仅观测位置）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # 过程噪声协方差
        q = process_noise ** 2
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float64)

        # 测量噪声协方差
        r = measurement_noise ** 2
        self.R = r * np.eye(self.m, dtype=np.float64)

        # 初始状态和协方差
        self.x = np.zeros(self.n, dtype=np.float64)
        self.P = np.eye(self.n, dtype=np.float64) * 100.0

        self._initialized = False

    def initialize(self, x: float, y: float):
        """用首个测量值初始化状态。"""
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(self.n, dtype=np.float64) * 10.0
        self._initialized = True

    def _cubature_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成容积采样点。

        CKF使用 2n 个容积点（n为状态维度），
        每个点权重 1/(2n)。

        Returns:
            chi: (n, 2n) 容积点矩阵
            w: (2n,) 权重向量
        """
        n = self.n
        # Cholesky 分解 P = S * S^T
        try:
            S = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # 协方差矩阵不正定时的回退
            S = np.linalg.cholesky(self.P + 1e-6 * np.eye(n))

        # 容积点: xi = x + S * [+1, -1] 的各维度组合
        xi = np.ones(n)
        chi = np.zeros((n, 2 * n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            chi[:, i] = self.x + S @ (np.sqrt(n) * e_i)
            chi[:, i + n] = self.x - S @ (np.sqrt(n) * e_i)

        w = np.ones(2 * n) / (2 * n)
        return chi, w

    def predict(self):
        """预测步。"""
        # 用容积点通过状态转移函数传播
        chi, w = self._cubature_points()

        # 线性状态转移: F @ chi
        chi_pred = self.F @ chi  # (n, 2n)

        # 加权均值: x_pred = sum_i w[i] * chi_pred[:, i]
        x_pred = chi_pred @ w  # (n,)

        # 加权协方差: P = Q + sum_i w[i] * (dx_i @ dx_i^T)
        dx = chi_pred - x_pred[:, None]  # (n, 2n)
        P_pred = self.Q + (dx * w) @ dx.T  # (n, n)

        self.x = x_pred
        self.P = P_pred

    def update(self, measurement: np.ndarray):
        """更新步。

        Args:
            measurement: [x, y] 测量值
        """
        if not self._initialized:
            self.initialize(measurement[0], measurement[1])
            return

        # 生成容积点
        chi, w = self._cubature_points()

        # 通过测量函数传播（线性: H @ chi）
        z_pred_points = self.H @ chi  # (m, 2n)

        # 测量预测均值
        z_pred = z_pred_points @ w  # (m,)

        # 新息协方差: Pzz = R + sum_i w[i] * (dz_i @ dz_i^T)
        dz = z_pred_points - z_pred[:, None]  # (m, 2n)
        Pzz = self.R + (dz * w) @ dz.T  # (m, m)

        # 交叉协方差: Pxz = sum_i w[i] * (dx_i @ dz_i^T)
        dx = chi - self.x[:, None]  # (n, 2n)
        Pxz = (dx * w) @ dz.T  # (n, m)

        # 卡尔曼增益
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 状态更新
        innovation = measurement - z_pred
        self.x = self.x + K @ innovation
        self.P = self.P - K @ Pzz @ K.T

    def filter_sequence(self, measurements: np.ndarray) -> np.ndarray:
        """对整个测量序列做滤波。

        Args:
            measurements: (N, 2) 测量序列 [x, y]

        Returns:
            (N, 2) 滤波后的平滑序列
        """
        if len(measurements) == 0:
            return measurements

        result = np.zeros_like(measurements)
        for i in range(len(measurements)):
            self.predict()
            self.update(measurements[i])
            result[i] = self.x[:2]

        return result


def apply_ckf_filter(px_seq: np.ndarray, py_seq: np.ndarray,
                     dt: float = 1.0 / 71.66) -> Tuple[np.ndarray, np.ndarray]:
    """对像素坐标序列应用CKF滤波。

    Args:
        px_seq: X方向像素坐标序列
        py_seq: Y方向像素坐标序列
        dt: 时间步长

    Returns:
        (smoothed_px, smoothed_py) 滤波后的坐标序列
    """
    measurements = np.column_stack([px_seq, py_seq]).astype(np.float64)
    ckf = CKFFilter(dt=dt, process_noise=0.5, measurement_noise=1.0)
    smoothed = ckf.filter_sequence(measurements)
    return smoothed[:, 0], smoothed[:, 1]
