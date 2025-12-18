import pandas as pd
from torch import optim, nn
from torch.utils.data import DataLoader
from KET_Unet.KET_Unet import KET_UNet
from tools.data_pre import *
import torch
from tools.utils import print_epoch_picture


def save_checkpoint(flag, net, epoch, i, train_loss, train_losses, image, segment_image, out_image, save_path, weight_path, arg):
    if flag == 1:
        k = 50
    else:
        k = 100
    if i % k == 0:
        print(f"The {flag} st time of training, Epoch {epoch}, Batch {i}, Loss: {train_loss.item():.6f}")
        train_losses.append(train_loss.item())
        train_losses.append(train_loss.item())
        print_epoch_picture(image, segment_image, out_image, save_path, epoch, i, flag)
        torch.save(net, os.path.join(weight_path, f"_{flag}_{epoch}_{i}.pth"))



def train_data(arg, net, data_loader, device, criterion, opt, train_losses, save_path, weight_path, flag, patience=5):
    if flag == 1:
        num_epochs = args.num_epochs_1
    else:
        num_epochs = args.num_epochs_2


    for epoch in range(num_epochs):
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = criterion(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            # 记录
            save_checkpoint(flag, net, epoch, i, train_loss, train_losses, image, segment_image, out_image, save_path, weight_path, arg)


def train(args):
    # path
    weight_path = os.path.join("model_save", f"--{args.time}--{args.model_name}--", "params")
    save_path = os.path.join("model_save", f"--{args.time}--{args.model_name}--", "train_process")
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 初始化列表来存储训练损失
    train_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.model_name == 'KET_UNet':
        net = KET_UNet(1, 1).to(device)
    else:
        print('please choose correct model name')


    ## parameters
    opt = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.L1Loss()



    data_loader = DataLoader(MyDataset(os.path.join("data_save", "First_training_data")), batch_size=args.batch_size, shuffle=True)
    train_data(args, net, data_loader, device, criterion, opt, train_losses, save_path, weight_path, flag=1)
    data_loader = DataLoader(MyDataset(os.path.join("data_save", "Second_training_data")), batch_size=args.batch_size, shuffle=True)
    train_data(args, net, data_loader, device, criterion, opt, train_losses, save_path, weight_path, flag=2)
    df = pd.DataFrame(train_losses, columns=['Loss'])
    df.to_excel(os.path.join(save_path, "train_process.xlsx"), index=False)


if __name__ == '__main__':

    args = get_parameters("KET_UNet")
    model_file_path3 = train(args)

