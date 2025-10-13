import os
from collections import Counter

# 标签文件夹路径
label_dir = '/home/lvyong/BDD100K/bdd100k/labels/100k/val_txt'

# 用于存储所有出现的类别索引及其名称
class_indices = set()
class_names = []

# 遍历所有标签文件
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # 遍历每一行，提取类别索引
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_index = int(parts[0])
                class_indices.add(class_index)

# 将类别索引排序
sorted_class_indices = sorted(class_indices)

# 生成类别名称列表
for index in sorted_class_indices:
    class_names.append(f"class_{index}")

# 输出统计结果
print("数据集中出现的类别索引及其名称：")
for index, name in zip(sorted_class_indices, class_names):
    print(f"类别索引 {index}: {name}")

# 统计每个类别的出现次数
class_counter = Counter()
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # 遍历每一行，统计类别出现次数
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_index = int(parts[0])
                class_counter[class_index] += 1

# 输出每个类别的出现次数
print("\n每个类别的出现次数：")
for index, count in class_counter.items():
    print(f"类别索引 {index} ({class_names[index]}): {count} 次")