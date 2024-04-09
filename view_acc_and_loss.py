import pandas as pd
import matplotlib.pyplot as plt
from train_model import save_dir

"""
保存和可视化训练数据
已有csv文件后直接运行该模块即可查看数据
"""

acc_and_loss_dir = save_dir + '/acc_and_loss.csv'


def save_acc_and_loss(val_acc_history, train_acc_history, valid_losses, train_losses):
    # 创建 DataFrame
    df = pd.DataFrame({
        'val_acc': val_acc_history,
        'train_acc': train_acc_history,
        'val_loss': valid_losses,
        'train_loss': train_losses
    })

    # 保存到 CSV 文件
    df.to_csv(acc_and_loss_dir, index_label='epoch')


def draw_acc_and_loss():
    # 从 CSV 文件中读取数据
    df = pd.read_csv(acc_and_loss_dir)

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['val_acc'], 'b', label='Validation acc')
    plt.plot(df['epoch'], df['train_acc'], 'r', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_loss'], 'b', label='Validation loss')
    plt.plot(df['epoch'], df['train_loss'], 'r', label='Training loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 已有csv文件后直接运行该模块即可可视化
if __name__ == '__main__':
    draw_acc_and_loss()
