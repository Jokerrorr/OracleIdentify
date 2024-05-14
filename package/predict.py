from torchvision import transforms
from PIL import Image
import torch
import pickle

from resnet18.net import net


def predict(image_dir, topk, model, class_names_pkl_dir,
            device):  # image_dir传入需要预测的图路径,topk概率最高的k个,model初始化好的模型,class_names_pkl_dir类名称路径（pkl文件）,gpu or cpu
    # 加载图片
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
    image = Image.open(image_dir)
    image = transform(image)
    image = image.unsqueeze(0)  # 在第0维增加一个维度，代表批处理维度
    image = image.to(device)

    # 打开并读取分类名称
    with open(class_names_pkl_dir, 'rb') as f:
        class_names = pickle.load(f)

    # 识别并返回结果
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, prediction = torch.topk(outputs, topk)
        prediction_label = [(class_names[idx.item()], probabilities[idx].item()) for idx in prediction[0]]
        return prediction_label  # 返回预测的（label,probabilities）列表


if __name__ == '__main__':
    image_dir = '../resnet18/data/valid/60A0A/60A0A_12.png'

    topk = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = net(300)
    checkpoint_dir = "../resnet18/save_train_data/checkpoint.pth"
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    class_names_pkl_dir = "../resnet18/save_train_data/class_names.pkl"

    p = predict(image_dir, topk, model, class_names_pkl_dir, device)
    print(p)
