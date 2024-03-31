from torchvision import models
import torch.nn as nn


def set_parameter_requires_grad(model, requires_grad):
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False


def net(num_classes, requires_grad=False, use_pretrained=True):
    model = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model, requires_grad)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                             nn.LogSoftmax(dim=1))
    return model

