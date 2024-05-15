import sys
<<<<<<< HEAD
=======

>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31
sys.path.append('..')
from torchvision import models
import torch.nn as nn
from functools import partial
from Res2Net.Res2Net import res2net50_26w_4s

"""
模型
"""


# 替换实例的forward函数以便输出特征供NST蒸馏使用
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    l1 = self.layer1(x)
    l2 = self.layer2(l1)
    l3 = self.layer3(l2)
    l4 = self.layer4(l3)

    p_out = self.avgpool(l4)
    fea = p_out.view(p_out.size(0), -1)
    out = self.fc(fea)

    return l4, fea, out


# 冻结所有参数(fc层除外)
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def net(num_classes, use_pretrained=False):  # use_pretrained：是否使用预训练模型
    model = models.resnet18(pretrained=use_pretrained)
    model.forward = partial(forward, model)  # 替换forward
    set_parameter_requires_grad(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def TeacherNet(num_classes, use_pretrained=True):  # 教师模型,用于模型蒸馏
    model = res2net50_26w_4s(pretrained=use_pretrained)
    model.forward = partial(forward, model)  # 替换forward
    set_parameter_requires_grad(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
