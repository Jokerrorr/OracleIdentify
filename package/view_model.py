import torch
from torchviz import make_dot

"""
此处是展示模型，在模型训练结束后单独运行此项即可产生文件model和model.pdf
需要安装graphviz（安装教程：https://zhuanlan.zhihu.com/p/268532582）
直接运行可能出错，可以在终端中使用命令python view_model.py
"""


def view_model(save_model_dir, device, model_pt_dir):
    # 加载模型
    model = torch.load(model_pt_dir)

    # 输入样本数据
    sampledata = torch.randn(1, 3, 224, 224)  # 输入是224*224*3
    model.to(device)
    sampledata = sampledata.to(device)
    # 计算模型输出
    out = model(sampledata)

    # 创建计算图并保存为 PDF 文件
    g = make_dot(out)
    g.render(save_model_dir)


if __name__ == "__main__":
    save_model_dir = "../resnet18/save_train_data/model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_pt_dir = "../resnet18/save_train_data/model.pt"
    view_model(save_model_dir, device, model_pt_dir)
