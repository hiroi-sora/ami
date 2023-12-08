import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from img2vec_pytorch import Img2Vec
import os
import argparse
import sys

# 图片特征向量路径
VEC_PATH = "../cifar10_vec/"
# 工作路径改为脚本所在路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 命令行参数
parser = argparse.ArgumentParser(description="CIFAR-10 AMI experiments - noldp")
parser.add_argument("-r", "--numneurons", type=int, default=1000)
parser.add_argument("-o", "--output_path", type=str, default="res_cifar/")  # 结果输出路径
parser.add_argument("-s", "--seed", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print("Python 解释器路径: ", sys.executable)


# CIFAR-10数据集
class AMIADatasetCifar10(Dataset):
    def __init__(  # [50000], None, VEC_PATH
        self, target, transform, dataroot, train=True, imgroot=None, multiplier=100
    ):
        self.target = target  # 目标索引列表
        self.target_multiplier = multiplier  # 训练1 测试模式1000
        self.transform = transform  # 应用于图像的转换操作
        if train:  # 训练集模式
            self.train_data = np.arange(50000)  # 测试集数据的索引范围，0~49999
            # self.length = 1 * 1 + 50000 = 50001
            self.length = len(target) * multiplier + len(self.train_data)
        else:  # 测试集模式
            self.valid_data = np.arange(50001, 60000)  # 训练集数据的索引，50001~59999
            # self.length = 9999 + 1 * 1000 = 10999
            # 其中 9999 是非目标的干扰项， 1*1000 是目标。
            self.length = len(self.valid_data) + len(target) * multiplier
        self.dataroot = dataroot  # VEC_PATH 图片特征向量目录
        self.imgroot = imgroot  # None
        self.data_name = sorted(os.listdir(dataroot))
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 训练集模式
        if self.train:
            # 如果索引是目标 （即为0）
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[
                    self.target[int(idx / self.target_multiplier)]
                ]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                # 类别ID，为 0 真
                class_id = torch.tensor(int(idx / self.target_multiplier))
            # 如果索引不是目标（>0）
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                # 为 1 假
                class_id = torch.tensor(len(self.target))
        # 测试集模式
        else:
            if idx / self.target_multiplier < len(self.target):
                # 获取文件名
                filename = self.data_name[  # self.target[0]=50000
                    self.target[int(idx / self.target_multiplier)]
                ]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))

        if self.imgroot:  # None
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])  # √

        # img_tensor = img2vec.get_vec(img, tensor=True)
        # img_tensor = torch.squeeze(img_tensor)
        img_tensor = torch.load(self.dataroot + filename)

        # img_tensor = img_tensor + s1.astype(np.float32)
        # img_tensor为图片特征向量
        # class_id = 1(假，即不是目标) / 0(真，即是目标)     img=tensor([])
        return img_tensor, class_id, img


# 目标数量和目标列表
num_target = 1
target = [50000]
# 保存的文件名
SAVE_NAME = f"{args.output_path}/Cifar10_embed_{args.numneurons}_single_{target[0]}.pth"
print(SAVE_NAME)
print("Loading data...")

# 加载训练集和测试集，均为 preprocessing.py img2vec 预处理过的数据
# 即将图片输入 resnet-18 模型后， avgpool 层的输出。
train_loader = torch.utils.data.DataLoader(
    AMIADatasetCifar10(target, None, VEC_PATH, True, imgroot=None, multiplier=1),
    shuffle=False,  # 数据不会被打乱
    num_workers=0,  # 所有数据加载操作将在主进程中完成
    batch_size=200000,  # 每个批次加载的样本数量（实际上，就是将所有数据当成一个batch）
)
test_loader = torch.utils.data.DataLoader(
    AMIADatasetCifar10(target, None, VEC_PATH, False, imgroot=None, multiplier=1000),
    shuffle=False,
    num_workers=0,
    batch_size=200000,
)

x_train, y_train, _ = next(iter(train_loader))
x_train = x_train.to(device)  # Size([50001, 512])
y_train = y_train.to(device)  # Size([50001])

x_test, y_test, _ = next(iter(test_loader))
x_test = x_test.to(device)  # Size([10999, 512])
y_test = y_test.to(device)  # Size([10999])

# 打印标签张量中，不同元素的唯一值，及其出现的次数
# 唯一值： 0 1 出现次数： 1 50000  即目标1个
print(torch.unique(y_train, return_counts=True))
# 唯一值： 0 1 出现次数： 1000 9999  即目标1000个（multiplier=1000放大）
print(torch.unique(y_test, return_counts=True))
print("Done.")

# for reproducibility 设置随机种子以确保可复现性
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# 求 真正率（TPR）、真负率（TNR）、准确率
# prediction 和 truth ： 标签，每个元素为 T 或 F
def tpr_tnr(prediction, truth):
    # confusion_vector 混淆向量，表示每个预测值与真实值之间的混淆情况。对于每一项：
    # - prediction 0真 ， truth 0真 （真正例）： 结果为 NaN （因为0/0在PyTorch中为NaN）。
    # - prediction 1假 ， truth 1假 （真负例）： 结果为 1 （因为任何非零数字除以自身等于1）。
    # - prediction 1假 ， truth 0真 （假负例）： 结果为 inf （因为任何非零数字除以0等于无穷大）。
    # - prediction 0真 ， truth 1假 （假正例）： 结果为 0 （因为0除以任何非零数字等于0）。
    confusion_vector = prediction / truth

    # 真负例的数量。计算混淆向量中等于1的元素数
    true_negatives = torch.sum(confusion_vector == 1).item()
    # 假负例的数量。计算混淆向量中等于无穷大的元素数
    false_negatives = torch.sum(confusion_vector == float("inf")).item()
    # 真正例的数量。计算混淆向量中为NaN的元素数
    true_positives = torch.sum(torch.isnan(confusion_vector)).item()
    # 假正例的数量。计算混淆向量中等于0的元素数
    false_positives = torch.sum(confusion_vector == 0).item()

    # 计算真正率（TPR），即真正例的数量除以（真正例的数量加上假负例的数量）。
    tpr = true_positives / (true_positives + false_negatives)
    # 计算真负率（TNR），即真负例的数量除以（真负例的数量加上假正例的数量）。
    tnr = true_negatives / (true_negatives + false_positives)
    # 计算准确率，即（真正例的数量加上真负例的数量）除以总样本数（即所有四种情况的数量之和）。
    accuracy = (true_positives + true_negatives) / (
        true_negatives + false_negatives + true_positives + false_positives
    )

    # 返回一个元组，包含真正率、真负率和准确率。
    return (tpr, tnr, accuracy)


# eps=1: 2000 neurons
# eps=2: 1000 neurons
# 分类器模型
class Classifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        #  512 2
        super(Classifier, self).__init__()
        # 512 → 1000
        self.fc1 = nn.Linear(n_inputs, args.numneurons)
        # 1000 → 2 (目标个数+1)
        self.fc2 = nn.Linear(args.numneurons, n_outputs)
        # fc2输出： [50001, 2]

    def forward(self, x):
        x = torch.flatten(x, 1)  # 输入扁平化
        # 第一层
        x = self.fc1(x)
        x = F.relu(x)
        # 第二层
        fc2 = self.fc2(x)
        x = torch.sigmoid(fc2)
        probs = F.softmax(x, dim=1)
        # x: 第二层输出 , probs: sigmoid概率分布 , fc2: 第二层未激活的输出
        return x, probs, fc2


# 初始化模型和损失函数 512 2
model = Classifier(x_train.shape[1], num_target + 1)
model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)

# 设定交叉熵的权重，0真 25000 , 1假 0.1
# 模型对类别 0 的错误分类的惩罚会远远大于对类别 1 的错误分类。
custom_weight = np.array([25000, 0.1])
# 交叉熵损失函数对象
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(custom_weight, dtype=torch.float).to(device)
)

min_loss = 100000000000  # 初始设置一个很大的损失值
max_correct = 0  # 最大正确分类的数量
max_tpr = 0.0  # 最大 真正率
max_tnr = 0.0  # 最大 真负率
max_acc = 0.0  # 最大准确率
epoch = 0  # 当前的迭代次数

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for i in range(1000000):
    num_correct = 0  # 正确分类的样本数量
    num_samples = 0  # 总样本数量
    loss_value = 0  # 损失值
    epoch += 1  # 迭代次数增加
    model.train()  # 设置分类器模型为训练模式

    # 前向传播，传入训练集样本，计算 预测值、概率、fc2层的输出
    out, probs, fc2 = model(x_train)
    loss = criterion(out, y_train)  # 传入预测标签和训练标签，求损失值
    # 累加损失值
    loss_value += loss

    # 反向传播，计算梯度
    loss.backward()
    optimizer.step()  # 更新参数
    optimizer.zero_grad()  # 清空梯度

    # 提取 模型第二层线性输出 不激活(<0) 的样本
    predictions = fc2[:, 0] < 0
    # 传入神经元预测和训练标签，求 真正率（TPR）、真负率（TNR）
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)
    # x_train [50001, 512] , y_train [50001]
    # out [50001, 2] , fc2 [50001, 2] , predictions [50001]

    # 设置模型为评估模式
    model.eval()
    out, probs, fc2 = model(x_test)  # 测试集 推理
    predictions = fc2[:, 0] < 0
    tpr, tnr, _ = tpr_tnr(predictions, y_test)
    acc = (tpr + tnr) / 2  # 真正率和真负率的平均 - 准确率

    # 准确率超过了当前最佳数据，则保存一次参数
    if acc >= max_acc:
        state = {
            "net": model.state_dict(),
            "test": (tpr, tnr),
            "train": (tpr_train, tnr_train),
            "acc": acc,
            "lr": lr,
            "epoch": epoch,
        }

        max_acc = acc
        torch.save(state, SAVE_NAME)
        if acc == 1.0:  # 100%准确率
            break

        print(
            f"Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}"
        )

state = torch.load(SAVE_NAME)
print("Train: ", state["train"])
print("Test: ", state["test"])
print("Acc: ", state["acc"])
print("Epoch: ", state["epoch"])
