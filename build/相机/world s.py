import cv2
import numpy as np
import glob
from itertools import combinations
import os  # 用于提取文件名

# 标定板大小（内部角点数）
CHECKERBOARD = (5, 3)
# 角点优化的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 真实世界棋盘格坐标
square_size = 0.006  # 每个方格的边长（米）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 读取图片
# images = sorted(glob.glob('./00000.jpg'))
image_folder = './images002'  # 图片所在的文件夹
image_pattern = os.path.join(image_folder, '*.jpg')  # 匹配所有jpg文件
images = sorted(glob.glob(image_pattern))
# 存储棋盘格角点数据
objpoints = []  # 3D 坐标
imgpoints = []  # 2D 像素坐标

# 角点检测
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    else:
        print(f"未找到足够角点的图像: {fname}")

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 获取相机内参
f_x, f_y = mtx[0, 0], mtx[1, 1]
c_x, c_y = mtx[0, 2], mtx[1, 2]

# 遍历每张图片计算尺度因子
for img_idx in range(len(imgpoints)):
    pixel_points = imgpoints[img_idx]  # 该图像的所有角点
    rvec, tvec = rvecs[img_idx], tvecs[img_idx]

    # 计算旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    scale_factors = []

    # 遍历所有角点对
    for i, j in combinations(range(len(pixel_points)), 2):
        pixel_x_diff = pixel_points[j][0][0] - pixel_points[i][0][0]
        pixel_y_diff = pixel_points[j][0][1] - pixel_points[i][0][1]

        world_x_diff = objp[j][0] - objp[i][0]
        world_y_diff = objp[j][1] - objp[i][1]

        # 确保不会除以零
        if pixel_x_diff != 0 and pixel_y_diff != 0:
            numerator_x = (f_x * r11 + c_x * r31) * world_x_diff + (f_x * r12 + c_x * r32) * world_y_diff
            numerator_y = (f_y * r21 + c_y * r31) * world_x_diff + (f_y * r22 + c_y * r32) * world_y_diff

            scale_factor_x = numerator_x / pixel_x_diff
            scale_factor_y = numerator_y / pixel_y_diff

            # 取均值
            scale_factors.append((scale_factor_x + scale_factor_y) / 2)

    # 计算尺度因子的均值
    if scale_factors:
        average_scale_factor = np.mean(scale_factors)
        # 获取图片名称
        filename = os.path.basename(images[img_idx])  # 提取文件名部分
        print(f"图片 {filename} 的平均尺度因子: {average_scale_factor:.6f}")
