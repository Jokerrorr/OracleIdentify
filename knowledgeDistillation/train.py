import os
import pickle

import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets

from net import net, TeacherNet
from train_model import train_model, save_dir
from package.view_acc_and_loss import *
from view_distillation_acc import view_distillation_acc_st
from nst import NST

"""
用于训练的主要脚本
"""

epochs = 10
batch_size = 16
num_classes = 100
learning_rate = 1e-3

filename_s = save_dir + '/checkpoint_s.pth'  # 学生模型保存文件
filename_t = save_dir + '/checkpoint_t.pth'  # 教师模型保存文件
filename = save_dir + "/checkpoint.pth"  # 普通模型保存结果
# NST知识蒸馏参数
lambda_kd = 1.0  # 蒸馏loss的比重

# IsTeacher为True训练教师模型, False训练学生模型
# UseTeacher为True使用教师模型知识蒸馏学生模型, 否则不启用
IsTeacher = False
UseTeacher = True

<<<<<<< HEAD
=======

>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31
def train():
    # 数据路径
    data_dir = 'data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    filename = filename_t if IsTeacher else filename_s

    # 初始化数据
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(15),
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
    with open(save_dir + '/class_names.pkl', 'wb') as f:
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
    model = TeacherNet(num_classes) if IsTeacher else net(num_classes)
    # GPU计算
    model = model.to(device)
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
<<<<<<< HEAD
    scheduler = None
=======
>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31

    # 设置教师model
    model_t = None
    criterionKD = None
    if UseTeacher:
        try:
            checkpoint_t = torch.load(filename_t)
        except:
            raise RuntimeError('Not Found pth file of teacher model')
        model_t = TeacherNet(num_classes)
        try:
            model_t.load_state_dict(checkpoint_t['state_dict'])
        except:
            raise RuntimeError('Teacher model loading failed')
        model_t = model_t.to(device)
        # 设置NST知识蒸馏损失
        criterionKD = NST()

    """
    在此处设置需要更新梯度的参数,如果不设置则只会更新fc层
    """
    model.layer1.requires_grad_()
    if not IsTeacher:
        model.layer2.requires_grad_()
        model.layer3.requires_grad_()
        model.layer4.requires_grad_()

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
                                                                                        scheduler=cosine_schedule,
                                                                                        filename=filename,
                                                                                        model_t=model_t,
                                                                                        criterionKD=criterionKD,
                                                                                        lambda_kd=lambda_kd)
    # 绘图
    if IsTeacher:
        save_acc_and_loss_dir = save_dir + '/acc_and_loss_T.csv'
    elif UseTeacher:
        save_acc_and_loss_dir = save_dir + '/acc_and_loss_S.csv'
    else:
        save_acc_and_loss_dir = save_dir + '/acc_and_loss_SnoT.csv'
    save_acc_and_loss(val_acc_history, train_acc_history, valid_losses, train_losses, save_acc_and_loss_dir)
    draw_acc_and_loss(save_acc_and_loss_dir)
<<<<<<< HEAD
=======

>>>>>>> 9633feafb276d18ae44feceff791806f83d50d31

if __name__ == '__main__':
    train()
