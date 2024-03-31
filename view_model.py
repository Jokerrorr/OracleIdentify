import torch
from torchviz import make_dot
from net import net
from train import num_classes, filename

"""
此处是展示模型，在模型训练结束后单独运行此项即可产生文件model和model.pdf
需要安装graphviz（安装教程：https://zhuanlan.zhihu.com/p/268532582）
直接运行可能出错，可以在终端中使用命令python view_model.py
"""

# 加载模型
model = net(num_classes=num_classes)

# 加载检查点
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['state_dict'])

# 创建示例输入
example_input = torch.randn(1, 3, 224, 224)

# 可视化模型
output = model(example_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("model")
