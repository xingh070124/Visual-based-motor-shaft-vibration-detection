import pandas as pd

# 输入文件路径和输出文件路径
input_file = "outputleft down.xlsx"  # 替换为你的输入文件路径
output_file = "outputleft down.xlsx"  # 输出文件路径，修改文件名以避免覆盖原文件

# 读取 Excel 文件
data = pd.read_excel(input_file)

# 获取 B列和 C列的列名（假设 B 列和 C 列分别是第二列和第三列）
b_column = data.columns[1]  # B列的列名
c_column = data.columns[2]  # C列的列名

# 对 B列和 C列的所有数据减去基准值
data[b_column] = data[b_column] +0

data[c_column] = data[c_column] +1.02436125638347

# 保存到新的输出文件中
data.to_excel(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")