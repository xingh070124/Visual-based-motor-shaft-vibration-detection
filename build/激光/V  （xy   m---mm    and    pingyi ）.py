import pandas as pd

# 输入文件路径和输出文件路径
input_file = "9righ down.xlsx"  # 替换为你的输入文件路径
output_file = "9righ down.xlsx"  # 替换为你的输出文件路径

# 读取 Excel 文件
data = pd.read_excel(input_file)  # 假设表头在第一行

# 获取第二行（索引为1）作为基准值
base_row = data.iloc[0]

# 获取 B列和 C列的列名
b_column = data.columns[1]  # B列的列名
c_column = data.columns[2]  # C列的列名

# 对 B列和 C列的所有数据减去基准值（从第二行开始，索引为1）
# data[b_column] = data[b_column].sub(base_row[b_column])  # B列数据减去 B2的值
# data[c_column] = data[c_column].sub(base_row[c_column])  # C列数据减去 C2的值
data[b_column] = pd.to_numeric(data[b_column], errors='coerce') * 1
data[c_column] = pd.to_numeric(data[c_column], errors='coerce') *1.3
data.to_excel(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")


