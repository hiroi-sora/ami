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

# 结果输出路径
VEC_PATH = "../cifar10_vec/"
# 工作路径改为脚本所在路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 命令行参数
parser = argparse.ArgumentParser(description="CIFAR-10 AMI experiments - noldp")
parser.add_argument("-r", "--numneurons", type=int, default=1000)
parser.add_argument("-o", "--output_path", type=str, default="res_cifar/")
parser.add_argument("-s", "--seed", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


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


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print("Python 解释器路径: ", sys.executable)


# CIFAR-10数据集
class AMIADatasetCifar10(Dataset):
    def __init__(
        self, target, transform, dataroot, train=True, imgroot=None, multiplier=100
    ):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        if train:
            # self.valid_data = np.arange(50000, 60000)
            self.valid_data = np.arange(50000)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            # self.train_data = np.arange(50000)
            self.train_data = np.arange(50001, 60000)
            # mask = np.ones(50000, dtype=bool)
            # mask[target] = False
            # self.train_data = self.train_data[mask, ...]
            self.length = len(self.train_data) + len(target) * multiplier
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train == False:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[
                    self.target[int(idx / self.target_multiplier)]
                ]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))

        else:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[
                    self.target[int(idx / self.target_multiplier)]
                ]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))

        if self.imgroot:
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])

        # img_tensor = img2vec.get_vec(img, tensor=True)
        # img_tensor = torch.squeeze(img_tensor)
        img_tensor = torch.load(self.dataroot + filename)

        # img_tensor = img_tensor + s1.astype(np.float32)

        return img_tensor, class_id, img


# eps=1: 2000 neurons
# eps=2: 1000 neurons
# 分类器模型
class Classifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, args.numneurons)
        self.fc2 = nn.Linear(args.numneurons, n_outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        fc2 = self.fc2(x)
        x = torch.sigmoid(fc2)
        probs = F.softmax(x, dim=1)
        return x, probs, fc2


# 目标数量和目标列表
num_target = 1
target = [50000]
# 保存的文件名
SAVE_NAME = f"{args.output_path}/Cifar10_embed_{args.numneurons}_single_{target[0]}.pth"
print(SAVE_NAME)
print("Loading data...")

# 加载训练集和测试集
train_loader = torch.utils.data.DataLoader(
    AMIADatasetCifar10(target, None, VEC_PATH, True, imgroot=None, multiplier=1),
    shuffle=False,
    num_workers=0,
    batch_size=200000,
)
test_loader = torch.utils.data.DataLoader(
    AMIADatasetCifar10(target, None, VEC_PATH, False, imgroot=None, multiplier=1000),
    shuffle=False,
    num_workers=0,
    batch_size=200000,
)

x_train, y_train, _ = next(iter(train_loader))
x_train = x_train.to(device)
y_train = y_train.to(device)

x_test, y_test, _ = next(iter(test_loader))
x_test = x_test.to(device)
y_test = y_test.to(device)


# x_train = torch.load(f'{args.output_path}/Cifar_x_train.pt').to(device)
# y_train = torch.load(f'{args.output_path}/Cifar_y_train.pt').to(device)

# x_test = torch.load('Cifar-res/Cifar_x_test.pt').to(device)
# y_test = torch.load('Cifar-res/Cifar_y_test.pt').to(device)

print(torch.unique(y_train, return_counts=True))
print(torch.unique(y_test, return_counts=True))

print("Done.")

# for reproducibility 设置随机种子以确保可复现性
torch.manual_seed(args.seed)
np.random.seed(args.seed)


import sklearn.utils.class_weight

# 初始化模型和损失函数
model = Classifier(x_train.shape[1], num_target + 1)
model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)

custom_weight = np.array([25000, 0.1])
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(custom_weight, dtype=torch.float).to(device)
)

min_loss = 100000000000
max_correct = 0
max_tpr = 0.0
max_tnr = 0.0
max_acc = 0.0
epoch = 0

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

# 训练模型
for i in range(1000000):
    num_correct = 0
    num_samples = 0
    loss_value = 0
    epoch += 1

    # for imgs, labels in iter(train_loader):
    model.train()

    out, probs, fc2 = model(x_train)
    loss = criterion(out, y_train)

    loss_value += loss

    loss.backward()
    optimizer.step()  # make the updates for each parameter
    optimizer.zero_grad()  # a clean up step for PyTorch

    predictions = fc2[:, 0] < 0
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)

    # Test acc
    model.eval()
    out, probs, fc2 = model(x_test)
    predictions = fc2[:, 0] < 0
    tpr, tnr, _ = tpr_tnr(predictions, y_test)
    acc = (tpr + tnr) / 2

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
        if acc == 1.0:
            break

    if i % 1 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(
            f"Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}"
        )

#     if epoch % 1 == 0:
#         state = {
#             'net': model.state_dict(),
#             'test': (tpr, tnr),
#             'train': (tpr_train, tnr_train),
#             'acc' : acc,
#             'lr' : lr,
#             'epoch' : epoch
#         }

# #         max_tpr = (tpr + tnr)/2
#         torch.save(state, SAVE_NAME + '-epoch' + str(epoch))


print("Train: ", torch.load(SAVE_NAME)["train"])
print("Test: ", torch.load(SAVE_NAME)["test"])
print("Acc: ", torch.load(SAVE_NAME)["acc"])
print("Epoch: ", torch.load(SAVE_NAME)["epoch"])
