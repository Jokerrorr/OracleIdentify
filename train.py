import os
import pickle

import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets

from net import net
from train_model import train_model, save_dir
from view_acc_and_loss import *

"""
用于训练的主要脚本
"""

epochs = 10
batch_size = 64
num_classes = 100
learning_rate = 1e-2
filename = save_dir + '/checkpoint.pth'  # 模型保存文件

if __name__ == '__main__':
    # 数据路径
    data_dir = 'data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # 初始化数据
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),
                                     transforms.Resize(224),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ]),
        'valid': transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    # 使用'rb'模式打开文件以读取二进制数据
    with open(save_dir+'/class_names.pkl', 'wb') as f:
        # 使用pickle.dump()来存储列表
        pickle.dump(class_names, f)

    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化model
    model = net(num_classes)
    # GPU计算
    model = model.to(device)
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
    criterion = nn.NLLLoss()

    """
    在此处设置需要更新梯度的参数,如果不设置则只会更新fc层
    """
    model.layer1.requires_grad_()

    # 输出当前正在训练的参数名称
    params_to_update = []
    print('-' * 10)
    print('当前正在更新：')
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(name)
    print(params_to_update)
    # 开始训练
    model, val_acc_history, train_acc_history, valid_losses, train_losses = train_model(model=model,
                                                                                        dataloaders=dataloaders,
                                                                                        criterion=criterion,
                                                                                        optimizer=optimizer,
                                                                                        num_epochs=epochs,
                                                                                        device=device,
                                                                                        scheduler=scheduler,
                                                                                        filename=filename)

    # 绘图
    save_acc_and_loss(val_acc_history, train_acc_history, valid_losses, train_losses)
    draw_acc_and_loss()
