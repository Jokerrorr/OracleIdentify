import torch
from torchviz import make_dot
from train_model import model_pt_dir, save_dir

"""
此处是展示模型，在模型训练结束后单独运行此项即可产生文件model和model.pdf
需要安装graphviz（安装教程：https://zhuanlan.zhihu.com/p/268532582）
直接运行可能出错，可以在终端中使用命令python view_model.py
"""

model_dir = save_dir + '/model'
# 加载模型
model = torch.load(model_pt_dir)

# 输入样本数据
sampledata = torch.randn(1, 3, 224, 224)  # 假设 input_size 是输入特征的维度
model.to('cuda')
sampledata = sampledata.to('cuda')
# 计算模型输出
out = model(sampledata)

# 创建计算图并保存为 PDF 文件
g = make_dot(out)
g.render(model_dir)
