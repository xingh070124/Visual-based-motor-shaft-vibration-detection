"""激光位移传感器数据后处理模块。

合并自原始脚本 yuanqiujie.py 和 V (xy m---mm and pingyi).py。

对激光位移传感器采集的 Excel 数据，按指定的偏移量和缩放因子
处理 B 列和 C 列数据，支持不同实验工况的参数配置。
"""

import pandas as pd
from typing import Optional


def process_laser_data(
    input_file: str,
    output_file: Optional[str] = None,
    b_offset: float = 0.0,
    c_offset: float = 0.0,
    b_scale: float = 1.0,
    c_scale: float = 1.0,
) -> pd.DataFrame:
    """处理激光位移传感器的 Excel 数据。

    对 B 列和 C 列执行：new_value = (old_value + offset) * scale

    Args:
        input_file: 输入 Excel 文件路径
        output_file: 输出 Excel 文件路径，None 表示覆盖原文件
        b_offset: B 列偏移量
        c_offset: C 列偏移量
        b_scale: B 列缩放因子
        c_scale: C 列缩放因子

    Returns:
        处理后的 DataFrame

    使用示例：
        >>> # yuanqiujie.py 的参数
        >>> df = process_laser_data("outputleft down.xlsx",
        ...     b_offset=0.0, c_offset=1.02436125638347)

        >>> # V (xy...).py 的参数
        >>> df = process_laser_data("9righ down.xlsx",
        ...     b_scale=1.0, c_scale=1.3)
    """
    data = pd.read_excel(input_file)

    # 获取 B 列和 C 列的列名（假设分别是第 2 列和第 3 列）
    b_column = data.columns[1]
    c_column = data.columns[2]

    # 确保数值类型
    data[b_column] = pd.to_numeric(data[b_column], errors='coerce')
    data[c_column] = pd.to_numeric(data[c_column], errors='coerce')

    # 应用偏移和缩放
    data[b_column] = (data[b_column] + b_offset) * b_scale
    data[c_column] = (data[c_column] + c_offset) * c_scale

    # 保存
    if output_file is None:
        output_file = input_file
    data.to_excel(output_file, index=False)

    print(f"处理完成，结果已保存到 {output_file}")
    print(f"  B 列: +{b_offset}, ×{b_scale}")
    print(f"  C 列: +{c_offset}, ×{c_scale}")

    return data


def process_laser_data_relative(
    input_file: str,
    output_file: Optional[str] = None,
    b_scale: float = 1.0,
    c_scale: float = 1.0,
    base_row_index: int = 0
) -> pd.DataFrame:
    """以基准行为参考，处理激光数据（相对变换模式）。

    Args:
        input_file: 输入 Excel 文件路径
        output_file: 输出文件路径
        b_scale: B 列缩放因子
        c_scale: C 列缩放因子
        base_row_index: 基准行索引，默认 0（第一行）

    Returns:
        处理后的 DataFrame
    """
    data = pd.read_excel(input_file)

    b_column = data.columns[1]
    c_column = data.columns[2]

    data[b_column] = pd.to_numeric(data[b_column], errors='coerce') * b_scale
    data[c_column] = pd.to_numeric(data[c_column], errors='coerce') * c_scale

    if output_file is None:
        output_file = input_file
    data.to_excel(output_file, index=False)

    print(f"处理完成，结果已保存到 {output_file}")
    return data
