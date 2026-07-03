"""文件工具：自然排序、图像加载、路径查找。"""

import os
import re
import glob
from typing import List


def natural_key(s: str) -> list:
    """自然排序键函数，将字符串中的数字按数值排序。

    示例：
        >>> sorted(['frame10.png', 'frame2.png'], key=natural_key)
        ['frame2.png', 'frame10.png']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def find_images(folder: str, pattern: str = "*.png", sort_natural: bool = True) -> List[str]:
    """在文件夹中查找图像文件并排序。

    Args:
        folder: 图像文件夹路径
        pattern: glob 匹配模式，默认 "*.png"
        sort_natural: 是否使用自然排序，默认 True

    Returns:
        排序后的图像文件路径列表
    """
    image_paths = glob.glob(os.path.join(folder, pattern))
    if sort_natural:
        image_paths = sorted(image_paths, key=natural_key)
    else:
        image_paths = sorted(image_paths)
    return image_paths


def extract_filename(path: str, with_ext: bool = True) -> str:
    """从完整路径中提取文件名。

    Args:
        path: 文件路径
        with_ext: 是否保留扩展名，默认 True

    Returns:
        文件名（含或不含扩展名）
    """
    basename = os.path.basename(path)
    if with_ext:
        return basename
    return os.path.splitext(basename)[0]
