from PIL import Image
from img2vec_pytorch import Img2Vec
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path

print("开始")

# 初始化img2vec，使用CUDA加速
img2vec = Img2Vec(cuda=True)  # model="resnet-18"

# =============== CelebA数据集处理 ===============
IMG2VEC_PATH = "data_celebA/celeba_img2vec_resnet/"  # 定义存储图像向量的路径
IMG_PATH = "data_celebA/celeba/img_align_celeba/"  # 定义原始图像数据的路径

# 确保向量存储路径存在
Path(IMG2VEC_PATH).mkdir(parents=True, exist_ok=True)

# 获取图像文件名列表，并排序
data_name = sorted(os.listdir(IMG_PATH))

# 对每个图像文件名进行处理
for name in tqdm(data_name, unit="data"):
    img_loc = IMG_PATH + name  # 图像文件的完整路径
    img = Image.open(img_loc)  # 打开图像文件
    img_tensor = img2vec.get_vec(img, tensor=True)  # 将图像转换为向量
    img_tensor = torch.squeeze(img_tensor)  # 去除多余的维度
    torch.save(img_tensor, IMG2VEC_PATH + name)  # 保存图像向量
print("CelebA数据集 处理完毕")

# =============== CIFAR-10数据集处理 ===============
# 下载并加载CIFAR-10训练和验证数据集
train_data = datasets.CIFAR10("data_cifar10", train=True, download=True)
valid_data = datasets.CIFAR10("data_cifar10", train=False, download=True)
IMG2VEC_PATH = "cifar10_vec/"  # 定义存储CIFAR-10图像向量的路径
Path(IMG2VEC_PATH).mkdir(parents=True, exist_ok=True)

# 处理训练数据
data_t = list(train_data)
for i in tqdm(range(len(data_t))):
    img_tensor = img2vec.get_vec(data_t[i][0], tensor=True)  # 转换图像为向量
    img_tensor = torch.squeeze(img_tensor)  # 移除张量中维度大小为 1 的维度
    # 保存图像向量，文件名用0填充至5位数
    torch.save(img_tensor, IMG2VEC_PATH + str(i).zfill(5) + ".vec")

# 处理验证数据
data_t = list(valid_data)
for i in tqdm(range(len(data_t))):
    img_tensor = img2vec.get_vec(data_t[i][0], tensor=True)  # 转换图像为向量
    img_tensor = torch.squeeze(img_tensor)  # 去除多余的维度
    # 保存图像向量，文件名用0填充至5位数，并加上50000以区分
    torch.save(img_tensor, IMG2VEC_PATH + str(i + 50000).zfill(5) + ".vec")
print("CelebA数据集 处理完毕")

# =============== ImageNet数据集处理 ===============

IMG2VEC_PATH = "imagenette2/imagenette_img2vec_resnet/"  # 定义存储ImageNet图像向量的路径
IMG_PATH = "imagenette2/"  # 定义原始图像数据的路径
Path(IMG2VEC_PATH).mkdir(parents=True, exist_ok=True)

# 导入glob库，用于路径模式匹配
import glob

# 生成数据集文件名列表
data_name = []
start = len(IMG_PATH)
# 获取训练集图像文件路径
for f in glob.glob(IMG_PATH + "train/*/*.JPEG", recursive=True):
    data_name.append(f[start:])
# 获取验证集图像文件路径
for f in glob.glob(IMG_PATH + "val/*/*.JPEG", recursive=True):
    data_name.append(f[start:])

# 对每个图像文件名进行处理
for name in tqdm(data_name):
    img_loc = IMG_PATH + name  # 图像文件的完整路径
    img = Image.open(img_loc).convert("RGB")  # 打开图像文件，并确保为RGB
    img_tensor = img2vec.get_vec(img, tensor=True)  # 将图像转换为向量
    img_tensor = torch.squeeze(img_tensor)  # 去除多余的维度
    # 确保向量存储路径的父目录存在
    Path(IMG2VEC_PATH + name).parent.mkdir(parents=True, exist_ok=True)
    torch.save(img_tensor, IMG2VEC_PATH + name)  # 保存图像向量
print("ImageNet数据集 处理完毕")
