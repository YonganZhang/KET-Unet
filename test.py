import os
from datetime import datetime
import numpy as np
import openpyxl
from torch.utils.data import DataLoader

from KET_Unet.KET_Unet import KET_UNet
from tools.data_pre import MyDataset, get_parameters
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import peak_signal_noise_ratio


def test_main(args, model_file_path="model_save/--16点44分--UNetWithTransformer_my--/params/_2_199_0.pth"):
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间字符串
    formatted_time = current_time.strftime("--%H--%M--")
    # 创建数据加载器
    test_loader = DataLoader(MyDataset(os.path.join("data_save", "test_data")), batch_size=1, shuffle=False) #batch_size=args.batch_size
    # 调用测试函数，这里假设您的测试函数是test(args)
    test(args, test_loader, formatted_time, model_file_path)


def test_save(outputs, save_dir, i):
    for batch_idx in range(outputs.size(0)):
        # 获取当前batch的输出，假设是灰度图像
        batch_output = outputs[batch_idx]  # shape: (channels, height, width)
        _image = batch_output.cpu().numpy()
        # 调整数组的维度顺序为 (height, width, channels)
        image_array = np.transpose(_image, (1, 2, 0))
        # 将浮点数数据转换为 uint8 类型
        image_array = (image_array * 255).astype(np.uint8)
        if image_array.shape[2] == 1:
            # 如果是1维数组，复制成3维数组（假设是灰度图像）
            image_array = np.tile(image_array, (1, 1, 3))
        # 创建Image对象
        img = Image.fromarray(image_array)
        # 保存图像到本地文件
        img.save(os.path.join(save_dir, f"{i}.png"))


def test_save_three(outputs, targets, inputs, in_out_path, i):
    for batch_idx in range(outputs.size(0)):
        # 获取当前batch的输出，假设是灰度图像
        batch_output = outputs[batch_idx]  # shape: (channels, height, width)
        batch_targets = targets[batch_idx]  # shape: (channels, height, width)
        batch_inout = inputs[batch_idx]  # shape: (channels, height, width)
        _image = batch_inout.cpu().numpy()
        _segment_image = batch_targets.cpu().detach().numpy()
        _out_image = batch_output.cpu().detach().numpy()
        # 垂直方向拼接
        combined_image = np.zeros((_image.shape[0], _image.shape[1], _image.shape[1] * 3))
        # 将三个图像按照指定方式合并到新的图像中
        combined_image[:, 0:_image.shape[1], :_image.shape[1]] = _image
        combined_image[:, 0:_image.shape[1], _image.shape[1]:_image.shape[1] * 2] = _segment_image
        combined_image[:, 0:_image.shape[1], _image.shape[1] * 2:] = _out_image
        # 调整数组的维度顺序为 (height, width, channels)
        image_array = np.transpose(combined_image, (1, 2, 0))
        # 将浮点数数据转换为 uint8 类型
        image_array = (image_array * 255).astype(np.uint8)
        if image_array.shape[2] == 1:
            # 如果是1维数组，复制成3维数组（假设是灰度图像）
            image_array = np.tile(image_array, (1, 1, 3))
        # 创建Image对象
        img = Image.fromarray(image_array)
        # 保存图像到本地文件
        img.save(os.path.join(in_out_path, f"{i}.png"))


def test(args, test_loader, formatted_time, model_file_path):
    """
    测试函数，对每个测试集运行模型进行检验
    """
    # 定义模型，这里假设您的模型定义在test函数外部
    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(args, model_file_path, device)

    # 测试输出
    file_name = f"{formatted_time}测试输出结果"
    out_path = os.path.join(os.path.dirname(os.path.dirname(model_file_path)), file_name)
    # 测试输入
    file_name = f"{formatted_time}测试输入结果"
    in_path = os.path.join(os.path.dirname(os.path.dirname(model_file_path)), file_name)
    # 两种测试输入
    file_name = f"{formatted_time}测试输入和输出结果"
    in_out_path = os.path.join(os.path.dirname(os.path.dirname(model_file_path)), file_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(in_path):
        os.makedirs(in_path)
    if not os.path.exists(in_out_path):
        os.makedirs(in_out_path)
    # 设置模型为评估模式
    model.eval()
    model = torch.load(model_file_path)
    model.eval()
    print("模型已加载")

    # 误差预设
    mse_sum = 0.0
    psnr_sum = 0.0
    mae_sum = 0.0
    num_samples = 0

    # 预测与实际值
    print("开始测试模型")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # 将 outputs 写入文件
            # 遍历每个batch
            test_save(outputs, out_path, i)
            test_save(inputs, in_path, i)
            test_save_three(outputs, targets, inputs, in_out_path, i)

            ## 计算误差
            # 进行后处理，将小于50的值设为0
            outputs = outputs * 255
            targets = targets * 255
            outputs2 = outputs.cpu().numpy()
            targets2 = targets.cpu().numpy()
            mse = np.mean((targets2 - outputs2) ** 2)

            # 计算均方误差（MSE）
            batch_mse = torch.mean((outputs - targets) ** 2).item()
            mse_sum += batch_mse

            # 计算峰值信噪比（PSNR）
            for j in range(inputs.size(0)):
                output_img = to_pil_image(outputs[j].cpu())
                target_img = to_pil_image(targets[j].cpu())
                psnr_value = peak_signal_noise_ratio(np.array(target_img), np.array(output_img))
                psnr_sum += psnr_value

            # 计算平均绝对误差（MAE）
            batch_mae = torch.mean(torch.abs(outputs - targets)).item()
            mae_sum += batch_mae

            num_samples += inputs.size(0)

        # 计算平均值
        avg_mse = mse_sum / num_samples
        avg_psnr = psnr_sum / num_samples
        avg_mae = mae_sum / num_samples

        excel_file = os.path.join(os.path.dirname(os.path.dirname(model_file_path)), f"{formatted_time}误差结果.xlsx")

        # 创建或加载现有的Excel工作簿
        wb = openpyxl.Workbook()
        sheet = wb.active

        # 在第一行添加标题
        sheet["A1"] = "Average MSE"
        sheet["B1"] = "Average PSNR"
        sheet["C1"] = "Average MAE"

        # 在第二行添加数据
        sheet["A2"] = avg_mse
        sheet["B2"] = avg_psnr
        sheet["C2"] = avg_mae

        # 保存Excel文件
        wb.save(excel_file)


def create_model(args, model_file_path, device):
    """
        根据参数定义模型
    """
    model_names = [
        'UNet',
        'IRUNet',
        'AttU_Net',
        'My_AttU_Net',
        'UNetWithTransformer',
        'KET_UNet'
    ]

    # 获取模型名称
    parent_dir = os.path.dirname(os.path.dirname(model_file_path))
    # 使用字符串的 split 方法根据 '--' 分割字符串
    parts = parent_dir.split('--')
    # 选择包含模型名称的部分，假设模型名称位于倒数第二个元素
    model_name = parts[-2]

    # 检查模型是否存在于可选列表中
    if model_name not in model_names:
        raise ValueError('Model name does not match any available models')

    # 根据模型名称创建对应的模型实例
    if model_name == 'KET_UNet':
        net = KET_UNet(1, 1).to(device)
    else:
        raise ValueError('Please choose a correct model name')

    return net


if __name__ == "__main__":
    args = get_parameters()
    test_main(args, model_file_path="model_save/--16点39分--KET_UNet--/params/_2_199_0.pth")
