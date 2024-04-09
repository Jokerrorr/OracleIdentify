import time
import copy
import torch
import os

"""
用于训练的函数
"""

save_dir = 'save_train_data'
model_pt_dir = save_dir + '/model.pt'


# 训练模型的主要函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, scheduler=None, filename=None):
    since = time.time()  # 记录开始时间
    best_acc = 0

    # 如果有checkpoint则加载模型
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.to(device)

    # 记录数据
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # 打印当前epoch和分隔符
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':  # 训练时更新权重
                        loss.backward()
                        optimizer.step()
                # 计算损失
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print(f'Time elapsed {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
            print(f'{phase} Loss:{round(epoch_loss, 4)} Acc:{round(epoch_acc.item(), 4)}')

            # 每个epoch看现在是不是比以前的模型更好，如果是则保存下来
            if phase == 'valid' and epoch_acc > best_acc:  # 以验证集的准确率为指标，越高越好
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)

            # 记录数据，后续用于绘图
            if phase == 'valid':
                val_acc_history.append(epoch_acc.cpu().numpy())
                valid_losses.append(epoch_loss)
                if scheduler:
                    scheduler.step(epoch_loss)

            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().numpy())
                train_losses.append(epoch_loss)

    # 训练结束
    time_elapsed = time.time() - since
    print(f'Training complete {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Acc: {best_acc}')

    # 训练完后用最好的一次当做模型最终的结果返回
    model.load_state_dict(best_model_wts)
    torch.save(model, model_pt_dir)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses
