import time

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from scipy import stats
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import read_IQA_data,read_IQA_data_Hyper
from utils.privacy import *
import math

# from torchviz import make_dot
import logging
logger = logging.getLogger()


class clientIQA_Hyper(Client):


    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []  # 保存费舍尔信息矩阵的历史记录，用于客户端参数个性化时的加权操作。
        self.l1_loss = torch.nn.L1Loss()
        # self.loss = nn.MSELoss()
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=args.learning_rate_decay_gamma)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.patch_size = args.patch_size
        self.patch_num = args.patch_num
        self.seed = args.seed
        self.best_result = [-1, -1, 0]  # srcc,plcc,epoch
        self.data_list = args.select_data_list
        self.max_name_length = max(len(name) for name in self.data_list)
        # 定义迭代的进度
        self.iteration_counter = 0  # 用于记录当前客户端的迭代进度
        self.result_folder = args.result_folder
        self.data_spilt_factor = args.data_spilt_factor
        self.Communication_num = 0

        #优化器
        self.weight_decay = 5e-4
        self.lr = 2e-5
        self.lrratio = 10
        paras = [{'params': self.model.head.parameters(), 'lr': self.lr * self.lrratio},
                 {'params': self.model.base.parameters(), 'lr': self.lr}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

    def train(self, is_selected):
        if is_selected:
            """
                    执行本地训练，确保每次只进行指定数量的迭代（iteration），而不是训练整个 epoch。               
            """
            trainloader = self.load_train_data()
            #创建迭代器
            if not hasattr(self, 'train_iter') or self.train_iter is None or self.Communication_num%self.data_spilt_factor==1:
                # 如果迭代器不存在，或者已经耗尽，创建新的迭代器
                self.train_iter = iter(trainloader)
            self.model.train()

            # 保存 loss 的历史记录
            self.avg_epoch_loss =[]

            start_time = time.time()

            max_local_epochs = self.local_epochs
            if self.train_slow:  # 如果客户端是“慢客户端”（self.train_slow 为 True），则使用较少的训练轮次。
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            # 客户端训练
                # 确定客户端训练数据的 1/5，并使用 math.ceil 向上取整，确保覆盖足够的训练数据
            dataset_size = len(trainloader.dataset)
            self.max_iterations_per_round = math.ceil(dataset_size / (self.data_spilt_factor * trainloader.batch_size))
            self.Communication_num +=1

            epoch_loss = 0  # 记录每个 epoch 的总 loss

            for step in range(max_local_epochs):

                self.sum_loss = 0  # 用于记录训练过程总的loss
                self.train_num = 0  # 用于记录训练了多少个样本
                # 使用 tqdm 作为进度条显示
                with tqdm(total=self.max_iterations_per_round, desc=f"Client {self.id} Training Progress") as pbar:
                    # 迭代训练数据
                    while self.iteration_counter < self.max_iterations_per_round:
                        try:
                            # 获取批量数据
                            x, y = next(self.train_iter)
                        except StopIteration:
                            self.train_iter = None
                            # print("______________________数据迭代器已耗尽______________________")
                            # 数据迭代器耗尽后，退出循环，而不是重新开始新的 epoch
                            break
                        x = torch.as_tensor(x.to(self.device)).requires_grad_(False)
                        y = torch.as_tensor(y.to(self.device)).requires_grad_(False)



                        # Building target network
                        paras = self.model(x)  # 'paras' contains the network weights conveyed to target network
                        model_target = TargetNet(paras).cuda()
                        for param in model_target.parameters():
                            param.requires_grad = False
                        # Quality prediction
                        pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net


                        loss = self.l1_loss(pred.squeeze(), y.float().detach())
                        # loss = self.l1_loss(output.squeeze(), y.float().detach())

                        self.solver.zero_grad()
                        loss.backward()
                        self.solver.step()
                        #累加当前batch的loss到epoch_loss中
                        epoch_loss += loss.item()
                        self.sum_loss += loss.item()*y.shape[0]
                        self.train_num += y.shape[0]
                        # 更新迭代计数器
                        self.iteration_counter += 1
                        # 更新进度条
                        pbar.update(1)
                        # else:break
                    # 完成本轮训练后，重置计数器
                    self.iteration_counter = 0
                # 计算每个 epoch 的平均 loss，并添加到 epoch_loss_history 中
                self.avg_epoch_loss = epoch_loss / self.max_iterations_per_round
                # self.loss_history.append(self.avg_epoch_loss)
                Epoch_step = f'- Local_Epoch:{step} ' if max_local_epochs!=1 else ''
                logger.info(
                    f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id}{Epoch_step} - Average Loss: {self.avg_epoch_loss:.4f}")

            # self.model.cpu()

            # Update optimizer
            if self.Communication_num%self.data_spilt_factor==0:
                lr = self.lr / pow(10, (self.Communication_num%self.data_spilt_factor // 6))
                if self.Communication_num%self.data_spilt_factor > 8:
                    self.lrratio = 1
                self.paras = [{'params': self.model.head.parameters(), 'lr': lr * self.lrratio},
                              {'params': self.model.base.parameters(), 'lr': self.lr} ]
                self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

                self.train_time_cost['num_rounds'] += 1
                self.train_time_cost['total_cost'] += time.time() - start_time



        # else:
        #     trainloader = self.load_train_data()
        #     # self.model.to(self.device)
        #     self.model.eval()
        #     # Compute FIM and its trace after training
        #     fim_trace_sum = 0
        #     for i, (x, y) in enumerate(trainloader):
        #         # Forward pass
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         outputs = self.model(x)
        #         # Negative log likelihood as our loss
        #         loss = F.mse_loss(outputs, y)
        #
        #         # Compute gradient of the negative log likelihood w.r.t. model parameters
        #         grads = grad(loss, self.model.parameters())
        #
        #         # Compute and accumulate the trace of the Fisher Information Matrix
        #         for g in grads:
        #             fim_trace_sum += torch.sum(g ** 2).detach()
        #
        #     # add the fisher log
        #     self.fim_trace_history.append(fim_trace_sum.item())

            # Evaluate on the client's test dataset
            # test_acc = self.evaluate()
            # print(f"Client {self.id}, Test Accuracy: {test_acc:.1f}, FIM-T value: {fim_trace_sum.item():.1f}")
            # print(f"FIM-T value change: {(self.fim_trace_history[-1] - (self.fim_trace_history[-2] if len(self.fim_trace_history) > 1 else 0)):.1f}")

    def load_train_data(self, batch_size=None):
        # 重写
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_IQA_data_Hyper(self.dataset, self.id, self.seed, is_train=True, selected_data_list=self.data_list,
                                   patch_size=self.patch_size, patch_num=self.patch_num)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True,num_workers=16)#, num_workers=16

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_IQA_data_Hyper(self.dataset, self.id, self.seed, is_train=False, selected_data_list=self.data_list,
                                  patch_size=self.patch_size, patch_num=self.patch_num)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False,num_workers=16)#,


    def set_parameters(self, model):
        #只替换特征提取层
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    # 重写
    def test_metrics(self):
        # 重写test_metrics
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)

        self.model.eval()
        self.model.train(False)
        pred_scores = []
        gt_scores = []
        test_num = 0

        with torch.no_grad():
            steps2 = 0
            iteration = 0
            for img, label in tqdm(testloaderfull,
                                   desc=f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id} - Local  testing progress"):
                # if iteration<50:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
                paras = self.model(img)
                model_target = TargetNet(paras).cuda()
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                steps2 += 1
                test_num += label.shape[0]
                iteration = iteration + 1
                # else:break
        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        return test_num, test_srcc, test_plcc

    # 重写
    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()
        train_num = 0
        losses = 0

        with torch.no_grad():
            iteration = 0
            for img, label in tqdm(trainloader,
                                   desc=f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id} - Calculating trainning lossed progress"):
                # if iteration <25:
                img = torch.as_tensor(img.to(self.device))
                label = torch.as_tensor(label.to(self.device))
                pred = self.model(img)
                loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
                losses += loss_qa * label.shape[0]
                train_num += label.shape[0]
                iteration = iteration + 1
                # else:break

        return train_num, losses

# 接收超网络生成的权重和偏置，预测图像的质量分数。
class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        # q = F.dropout(q)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q

# 利用分组卷积操作，实现每张图像的权重和偏置都不同。
class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """

    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2],
                                     self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])
    # def evaluate(self):
    #     # 评估客户端模型在测试集上的性能
    #     testloader = self.load_test_data()
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for x, y in testloader:
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             outputs = self.model(x)
    #             _, predicted = outputs.max(1)
    #             total += y.size(0)
    #             correct += predicted.eq(y).sum().item()
    #     accuracy = 100. * correct / total
    #     return accuracy