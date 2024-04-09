from torchvision import models
import torch.nn as nn

"""
模型
"""


# 冻结所有参数(fc层除外)
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def net(num_classes, use_pretrained=True):  # use_pretrained：是否使用预训练模型
    model = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                             nn.LogSoftmax(dim=1))
    return model
