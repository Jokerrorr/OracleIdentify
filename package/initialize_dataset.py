import os
import random
import shutil

"""
由于原始数据路径类似data/label/label_i.png,而我们需要将其分为训练集和验证集,即目标路径为data/train/label/label_i.png和data/valid/label/label_j.png
通过运行此文件，可以将原始数据按照一定比例分类为训练集和验证集
"""

# 数据集的根目录
data_root = '../data'

# 训练集占总数据集的比例
train_ratio = 0.8

# 获取所有类别的文件夹
classes = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

# 遍历每个类别
for class_name in classes:
    class_path = os.path.join(data_root, class_name)
    files = os.listdir(class_path)

    # 随机打乱文件列表
    random.shuffle(files)

    # 计算训练集和验证集中的文件数量
    train_count = int(len(files) * train_ratio)
    val_count = len(files) - train_count

    # 创建训练集和验证集的文件夹（如果不存在）
    train_dir = os.path.join(data_root, 'train', class_name)
    val_dir = os.path.join(data_root, 'valid', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 将文件移动到对应的训练集或验证集文件夹
    for i, file in enumerate(files):
        file_path = os.path.join(class_path, file)
        if i < train_count:
            shutil.move(file_path, train_dir)
        else:
            shutil.move(file_path, val_dir)
