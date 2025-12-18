import argparse
from datetime import datetime
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_parameters(modelname="UNet"):
    parser = argparse.ArgumentParser(description='训练模型的脚本')
    ## model
    parser.add_argument('--model_name', type=str, default=modelname, help='选择一个模型')
    parser.add_argument('--dropout', type=float, default=0.2, help='丢失概率')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%H点%M分"), help='用来存文件')

    # training
    parser.add_argument('--num_epochs_1', type=int, default=7, help='第一轮训练的轮数')
    parser.add_argument('--num_epochs_2', type=int, default=200, help='第二轮训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')

    # data
    parser.add_argument('--batch_size', type=int, default=6, help='批次大小')
    args = parser.parse_args()

    return args

import os
import re
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', s)]


class MyDataset(Dataset):
    def __init__(self, path):
        self.image_path = os.path.join(path, 'input')
        self.label_path = os.path.join(path, 'lable')

        self.name = sorted(os.listdir(self.label_path), key=natural_key)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        img_name = self.name[idx]

        image = Image.open(os.path.join(self.image_path, img_name)).convert('L')
        label = Image.open(os.path.join(self.label_path, img_name)).convert('L')

        image = self.transform(image)
        label = self.transform(label)

        return image, label



if __name__ == '__main__':
    #先建立独热编码对象
    # 可以使用 PIL 或其他库来保存或显示解码后的图像
    data = MyDataset('../data_save/Second_training_data')
    print(data[0][0].shape)
    print(data[0][1].shape)

