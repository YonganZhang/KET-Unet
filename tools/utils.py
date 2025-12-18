import numpy as np
from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def keep_image_size_open_rgb(path, size=(512, 512)):
    img = Image.open(path).convert('L')  # 转换为灰度图像
    temp = max(img.size)
    mask = Image.new('L', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def print_epoch_picture(image, segment_image, out_image, save_path, epoch, i, flag):
    _image = image[0].cpu().numpy()
    _segment_image = segment_image[0].cpu().detach().numpy()
    _out_image = out_image[0].cpu().detach().numpy()
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
    img.save(os.path.join(save_path, f"--{flag}--{epoch}--{i}--.png") )


def print_log(content, arg, log_file_path="output/"):
    # 确保日志文件夹存在
    log_file_path2 = os.path.join(log_file_path, arg.time, f"{arg.model_name}_log.txt")
    os.makedirs(os.path.dirname(log_file_path2), exist_ok=True)
    # 打开文件并写入内容
    with open(log_file_path2, 'a') as log_file:
        log_file.write(content + "\n")
