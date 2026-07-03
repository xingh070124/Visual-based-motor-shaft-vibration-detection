"""相机标定模块：张氏标定法，支持角点检测、标定、去畸变、重投影误差计算。

合并自原始脚本 cameraCalibration.py 和 world s.py。
"""

import os
import cv2
import numpy as np
import glob
from typing import List, Tuple, Optional


class CameraCalibrator:
    """张氏标定法相机标定器。

    使用棋盘格图像序列进行相机标定，获取内参、畸变系数和外参。

    Attributes:
        checkerboard_size: 棋盘格内角点数 (cols, rows)
        square_size: 方格边长（米），None 表示不使用真实世界尺寸
        criteria: 角点优化终止条件
        objpoints: 存储所有图像的 3D 角点坐标
        imgpoints: 存储所有图像的 2D 角点坐标
        mtx: 相机内参矩阵 (3×3)
        dist: 畸变系数
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        image_size: 图像尺寸 (w, h)
    """

    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: Optional[float] = None,
        criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        ),
        use_sb_detector: bool = True
    ):
        """初始化标定器。

        Args:
            checkerboard_size: 棋盘格内角点数 (cols, rows)
            square_size: 每个方格的边长（米），None 表示使用无单位坐标
            criteria: 角点优化终止条件
            use_sb_detector: True=使用 findChessboardCornersSB（更鲁棒），
                            False=使用传统 findChessboardCorners
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.criteria = criteria
        self.use_sb_detector = use_sb_detector

        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.image_paths: List[str] = []

        self.mtx: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.image_size: Optional[Tuple[int, int]] = None

        # 预计算 3D 世界坐标模板
        self._objp = self._build_object_points()

    def _build_object_points(self) -> np.ndarray:
        """构建棋盘格 3D 世界坐标模板。"""
        cols, rows = self.checkerboard_size
        objp = np.zeros((1, cols * rows, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        if self.square_size is not None:
            objp[0, :, :2] *= self.square_size
        return objp

    def detect_corners(
        self,
        image_folder: str,
        pattern: str = "*.jpg",
        show_progress: bool = False
    ) -> int:
        """从文件夹读取图像并检测棋盘格角点。

        Args:
            image_folder: 图像文件夹路径
            pattern: glob 匹配模式
            show_progress: 是否显示检测结果窗口

        Returns:
            成功检测到角点的图像数量
        """
        self.objpoints = []
        self.imgpoints = []
        self.image_paths = []

        images = sorted(glob.glob(os.path.join(image_folder, pattern)))

        for i, fname in enumerate(images):
            img = cv2.imread(fname)
            if img is None:
                print(f"[WARN]️ 无法读取图像: {fname}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.use_sb_detector:
                ret, corners = cv2.findChessboardCornersSB(
                    gray, self.checkerboard_size,
                    cv2.CALIB_CB_LARGER + cv2.CALIB_CB_MARKER
                )
            else:
                ret, corners = cv2.findChessboardCorners(
                    gray, self.checkerboard_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                    cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE
                )

            if ret:
                self.objpoints.append(self._objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                self.imgpoints.append(corners2)
                self.image_paths.append(fname)

                if show_progress:
                    img_disp = cv2.drawChessboardCorners(
                        img, self.checkerboard_size, corners2, ret
                    )
                    cv2.imshow(f"{fname} - succeed", img_disp)
                    cv2.waitKey(1)
            else:
                print(f"第 {i} 张图，{fname} 未发现足够角点")

        if show_progress:
            cv2.destroyAllWindows()

        return len(self.objpoints)

    def calibrate(self, flags: int = None) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """执行相机标定。

        必须先调用 detect_corners() 收集角点数据。

        Args:
            flags: cv2.calibrateCamera 的标定标志位，如 cv2.CALIB_FIX_K3。
                   None 时自动使用 CALIB_FIX_K3（固定 k3=0）。

        Returns:
            (mtx, dist, rvecs, tvecs): 内参矩阵、畸变系数、旋转向量、平移向量

        Raises:
            ValueError: 如果没有足够的角点数据
        """
        if len(self.objpoints) == 0:
            raise ValueError("没有角点数据，请先调用 detect_corners()")

        # 使用第一张图像确定尺寸
        sample_img = cv2.imread(self.image_paths[0])
        gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]  # (w, h)

        if flags is None:
            flags = cv2.CALIB_FIX_K3

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints,
            self.image_size, None, None,
            flags=flags
        )

        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        return mtx, dist, rvecs, tvecs

    def undistort(
        self,
        image: np.ndarray,
        alpha: float = 1.0,
        method: str = "undistort"
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """对图像进行去畸变处理。

        Args:
            image: 输入图像
            alpha: 自由缩放参数（0~1），1 表示保留所有像素
            method: "undistort" 或 "remap"

        Returns:
            (校正后图像, 裁剪后图像, ROI)
        """
        if self.mtx is None:
            raise ValueError("请先调用 calibrate() 获取内参")

        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha, (w, h)
        )

        if method == "remap":
            mapx, mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, new_camera_mtx, (w, h), 5
            )
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        else:
            dst = cv2.undistort(image, self.mtx, self.dist, None, new_camera_mtx)

        # 裁剪
        x, y, w_roi, h_roi = roi
        dst_cropped = dst[y:y + h_roi, x:x + w_roi]

        return dst, dst_cropped, roi

    def compute_reprojection_error(self) -> float:
        """计算重投影误差。

        Returns:
            平均重投影误差
        """
        if self.mtx is None:
            raise ValueError("请先调用 calibrate()")

        mean_error = 0.0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i],
                self.tvecs[i], self.mtx, self.dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(self.objpoints)

    def get_intrinsics_str(self) -> str:
        """获取格式化的内参信息字符串。"""
        if self.mtx is None:
            return "尚未标定"
        return (
            f"fx={self.mtx[0, 0]:.4f}, fy={self.mtx[1, 1]:.4f}, "
            f"cx={self.mtx[0, 2]:.4f}, cy={self.mtx[1, 2]:.4f}"
        )

    def print_calibration_result(self):
        """打印完整的标定结果到控制台。"""
        if self.mtx is None:
            print("尚未执行标定")
            return

        print("=" * 60)
        print("相机标定结果")
        print("=" * 60)
        print(f"棋盘格内角点: {self.checkerboard_size}")
        print(f"标定图像数量: {len(self.objpoints)}")
        print(f"\n相机内参:\n{self.mtx}")
        print(f"\n畸变参数 (k1, k2, p1, p2, k3):\n{self.dist.ravel()}")
        print(f"\n重投影误差: {self.compute_reprojection_error():.6f}")
        print("=" * 60)

    def get_rotation_matrices(self) -> List[np.ndarray]:
        """将所有旋转向量转换为旋转矩阵。

        Returns:
            旋转矩阵列表 (3×3)
        """
        if self.rvecs is None:
            raise ValueError("请先调用 calibrate()")
        return [cv2.Rodrigues(rvec)[0] for rvec in self.rvecs]

    def get_extrinsics_at(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定图像的旋转矩阵和平移向量。

        Args:
            index: 图像索引

        Returns:
            (rotation_matrix (3×3), tvec (3×1))
        """
        if self.rvecs is None:
            raise ValueError("请先调用 calibrate()")
        R, _ = cv2.Rodrigues(self.rvecs[index])
        return R, self.tvecs[index]
