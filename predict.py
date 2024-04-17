from torchvision import transforms
from PIL import Image
import torch
import pickle

from net import net
from train import filename, num_classes, save_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(save_dir + '/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)


def predict(image_dir):  # image_dir传入需要预测的图路径
    # 加载图片
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
    image = Image.open(image_dir)
    image = transform(image)
    image = image.unsqueeze(0)  # 在第0维增加一个维度，代表批处理维度
    image = image.to(device)

    # 加载模型
    model = net(num_classes)
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # 识别并返回结果
    with torch.no_grad():
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
        prediction_label = class_names[prediction]
        return prediction_label


if __name__ == '__main__':
    image_dir = 'data/valid/60A0A/60A0A_13.png'
    p = predict(image_dir)
    print(p)
