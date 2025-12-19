import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import random
import torchvision
from utils import folders
from scipy import stats
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



# 定义与训练时一致的模型结构
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

# 加载模型函数
def load_model(model_path):
    base_model = models.resnet50(pretrained=False)
    head_model = nn.Sequential(
        nn.Linear(base_model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1)
    )
    base_model.fc = nn.Identity()
    model = BaseHeadSplit(base_model, head_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model.to(device)  # 将模型移到GPU


def predict(model, data_loader,pfl_flag,datasetname):
    model.eval()
    predictions = []
    gt_scores = []
    pred_scores = []

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Predicting", unit="batch"):
            data = data.to(device).requires_grad_(False)
            pred = model(data)
            if (pfl_flag):
                target = torch.as_tensor(target.to(device)).requires_grad_(False)
            else:
                target = torch.as_tensor(remap_mos(datasetname,target.to(device))).requires_grad_(False)
            pred_scores.extend(pred.cpu().tolist())
            gt_scores.extend(target.cpu().tolist())

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 50)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 50)), axis=1)

    # 计算SRCC和PLCC
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    return pred_scores, gt_scores, test_srcc, test_plcc

    #将mos值进行重映射
def remap_mos(dataset_name,y):
        """
        根据数据集名称对MOS值进行线性重映射到[0, 100]区间
        """
        # 数据集对应的最小值和最大值
        dataset_min_max = {
            'tid2013': (0.24242, 7.21429),
            'kadid10k': (1, 4.93),
            'clive': (3.42, 92.43195266),
            'koniq': (3.91176470588235, 88.3888888888889),
            'csiq': (0, 1)
        }

        if dataset_name not in dataset_min_max:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        min_val, max_val = dataset_min_max[dataset_name]

        if dataset_name == 'csiq':
            # CSIQ 的 DMOS 需要取反映射，公式为 max_val - y
            y = max_val - y

        # 线性重映射公式
        return ((y - min_val) / (max_val - min_val)) * 100

# 计算预测误差
def calculate_error(predictions, targets):
    return np.abs(predictions - targets)  # 计算绝对误差

# 数据加载部分
def read_IQA_data(dataset, idx, seed=2021, is_train=True, selected_data_list=[], patch_size=(224, 224), patch_num=50):
    random.seed(seed)
    DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
    using_list = DATA_LIST if selected_data_list == [] else selected_data_list

    folder_path = {
        'live': os.path.join('/data/zyh26/PFL', dataset, 'LIVE/'),
        'csiq': os.path.join('/data/zyh26/PFL', dataset, 'CSIQ/'),
        'tid2013': os.path.join('/data/zyh26/PFL', dataset, 'TID2013/'),
        'kadid10k': os.path.join('/data/zyh26/PFL', dataset, 'kadid10k'),
        'clive': os.path.join('/data/zyh26/PFL', dataset, 'ChallengeDB_release'),
        'koniq': os.path.join('/data/zyh26/PFL', dataset, 'koniq-10k'),
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid10k': list(range(0, 80)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
    }

    total_num_images = img_num[using_list[idx]]
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]  # 80%训练集
    test_index = total_num_images[0:len(total_num_images)]  # 20%测试集

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])

    if using_list[idx] == 'koniq':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])

    path = folder_path[using_list[idx]]
    img_indx = test_index  # 使用测试集
    if using_list[idx] == 'csiq':
        data = folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num
        )
    elif using_list[idx] == 'clive':
        data = folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num
        )
    elif using_list[idx] == 'koniq':
        data = folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'tid2013':
        data = folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'kadid10k':
        data = folders.Kadid10k(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    else:
        raise ValueError(f"Unsupported dataset: {using_list[idx]}")

    return data

def plot_scatter(predictions_1, predictions_2, ground_truth, model_name_1, model_name_2):
    # 绘制模型1的散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(ground_truth, predictions_1, color='blue', alpha=0.5)
    # plt.title(f'{model_name_1} Predictions vs Ground Truth (CSIQ)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'r--')
    plt.tight_layout()

    # 保存模型1的散点图
    scatter_img_path_1 = f"/home/zyh26/225_FedAS-main/FedAS-main/test_img/{model_name_1}.png"
    plt.savefig(scatter_img_path_1)
    plt.show()

    # 绘制模型2的散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(ground_truth, predictions_2, color='green', alpha=0.5)
    # plt.title(f'{model_name_2} Predictions vs Ground Truth (CSIQ)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'r--')
    plt.tight_layout()

    # 保存模型2的散点图
    scatter_img_path_2 = f"/home/zyh26/225_FedAS-main/FedAS-main/test_img/{model_name_2}.png"
    plt.savefig(scatter_img_path_2)
    plt.show()

# 保存结果到CSV
def save_to_csv(predictions_1, predictions_2, ground_truth, model_name_1, model_name_2, csv_path):
    """
    保存模型预测结果和真实值到CSV文件。

    参数:
        predictions_1: 模型1的预测结果
        predictions_2: 模型2的预测结果
        ground_truth: 真实值
        model_name_1: 模型1的名称
        model_name_2: 模型2的名称
        csv_path: 保存CSV文件的路径
    """
    # 创建一个DataFrame来保存数据
    df = pd.DataFrame({
        'ground_truth': ground_truth,
        f'{model_name_1}_predictions': predictions_1,
        f'{model_name_2}_predictions': predictions_2
    })

    # 将数据保存到CSV文件
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

# 加载第一个模型路径
model_path_1 = "/data/zyh26/PFL_result/31_result/pfl_clive_clive_FedIQA_uniformEpoch_resnet_2e-05_5e-04_011316/client_0_clive.pt"  # 替换为模型路径
# 加载第二个模型路径
# model_path_2 = "/data/zyh26/PFL_result/Resnet/pfl_5_FedIQA_uniformEpoch_resnet_2e-05_5e-04_112809/run_9/client_0_csiq.pt"  # 替换为模型路径

# # 加载第一个模型路径
# model_path_1 = "/data/zyh26/PFL_result/Resnet/csiq_FedIQA_2e-05_5e-04_101010/run_9/client_0_csiq.pt"  # 替换为模型路径
# # 加载第二个模型路径
# model_path_2 = "/data/zyh26/PFL_result/Resnet/pfl_5_FedIQA_uniformEpoch_resnet_2e-05_5e-04_112809/run_9/client_0_csiq.pt"  # 替换为模型路径


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载两个模型
model_1 = load_model(model_path_1)
# model_2 = load_model(model_path_2)

# 读取测试集数据
dataset = 'IQA'  # 替换为数据集名称
DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
# idx = 2 # 选择你需要测试的子集（如 'csiq', 'live' 等）#'live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq'
# datasetname = DATA_LIST[idx]
# test_data = read_IQA_data(dataset, idx, is_train=False)
#
# # 创建 DataLoader
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False,num_workers=16)
#
# # 进行预测
# predictions_1, targets_1, srcc_1, plcc_1 = predict(model_1, test_loader,False,datasetname)
# # predictions_2, targets_2, srcc_2, plcc_2 = predict(model_2, test_loader,True)
#
# # 输出模型1和模型2的SRCC和PLCC
# print(f"Model 1 SRCC: {srcc_1}, PLCC: {plcc_1}")
# # print(f"Model 2 SRCC: {srcc_2}, PLCC: {plcc_2}")

for idx in range(1, 6):  # idx 从 1 遍历到 5
    datasetname = DATA_LIST[idx]

    if datasetname == 'clive':
        print(f"Skipping {datasetname} dataset...")
        continue  # 跳过当前循环

    test_data = read_IQA_data(dataset, idx, is_train=False)

    # 创建 DataLoader
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=16)

    # 进行预测
    predictions_1, targets_1, srcc_1, plcc_1 = predict(model_1, test_loader, False, datasetname)

    # 输出每个数据集的 SRCC 和 PLCC
    print(f"Dataset: {datasetname} | Model 1 SRCC: {srcc_1}, PLCC: {plcc_1}")

# 绘制散点图
# plot_scatter(predictions_1, predictions_2, targets_1, 'Model 1', 'Model 2')
# 保存路径
# csv_path = "/home/zyh26/225_FedAS-main/FedAS-main/test_img/results.csv"
# 保存结果到CSV
# save_to_csv(predictions_1, predictions_2, targets_1, 'Model 1', 'Model 2',csv_path)
