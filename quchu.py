
import os


def remove_class_and_save_to_new_folder(input_folder, output_folder, target_class):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 .txt 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # 读取文件内容
            with open(input_file_path, 'r') as f:
                lines = f.readlines()

            # 过滤掉类别为 target_class 的行
            new_lines = [line for line in lines if not line.strip().split()[0] == str(target_class)]

            # 将新内容写入到输出文件夹中的新文件
            with open(output_file_path, 'w') as f:
                f.writelines(new_lines)

            # 如果有删除操作，则打印文件名提示
            if len(new_lines) < len(lines):
                print(f"已删除文件 {filename} 中类别为 {target_class} 的目标，并保存到新文件夹")


# 使用示例
input_folder = '/home/lvyong/bdd100k/labels/100k/val_txt'  # 替换为你的标签文件夹路径
output_folder = '/home/lvyong/bdd100k/labels/100k/val_new'  # 替换为你想保存的输出文件夹路径
target_class = 9  # 要删除的目标类别
remove_class_and_save_to_new_folder(input_folder, output_folder, target_class)