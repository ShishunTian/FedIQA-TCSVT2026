import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from scipy import stats
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import read_IQA_data,read_IQA_data_All
from utils.privacy import *
import math

# from torchviz import make_dot
import logging
logger = logging.getLogger()


class clientIQA(Client):


    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []  # 保存费舍尔信息矩阵的历史记录，用于客户端参数个性化时的加权操作。
        self.l1_loss = torch.nn.L1Loss()
        # self.loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=args.learning_rate_decay_gamma)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.patch_size = args.patch_size
        self.patch_num = args.patch_num
        self.seed = args.seed
        self.best_result = [-1, -1, 0]  # srcc,plcc,epoch
        self.data_list = args.select_data_list
        self.dataset_name = self.data_list[self.id]
        self.pfl_flag = args.pfl
        self.normalization = args.normalization
        self.max_name_length = max(len(name) for name in self.data_list)
        # 定义迭代的进度
        self.iteration_counter = 0  # 用于记录当前客户端的迭代进度
        self.max_iterations_per_round = 50  # 每轮本地训练的最大迭代次数
        # if self.data_list[self.id]== 'kadid10k' or self.data_list[self.id]== 'koniq':
        #     self.max_iterations_per_round = self.max_iterations_per_round*4
        self.have_epoch = 0
        self.result_folder = args.result_folder
        self.data_spilt_factor = args.data_spilt_factor
        self.Communication_num = 0
        self.test_dataset =args.test_dataset
        self.pre_SRCC = 1

    def train(self, is_selected):
        if is_selected:
            """
                    执行本地训练，确保每次只进行指定数量的迭代（iteration），而不是训练整个 epoch。               
            """
            self.Communication_num += 1

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

            epoch_loss = 0  # 记录每个 epoch 的总 loss

            for step in range(max_local_epochs):
                # for i, (x, y) in enumerate(tqdm(trainloader,
                #                                 desc=f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id} - Local_Epoch:{step} - Training progress")):
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

                        ##根据是否为个性化联邦学习，判断是否需要重映射
                        if(self.normalization):
                            y = torch.as_tensor(self.remap_mos(y.to(self.device), self.dataset_name)).requires_grad_(
                                False)
                            # print("未进行重映射")
                        else:
                            y = torch.as_tensor(y.to(self.device)).requires_grad_(False)

                            # print("进行重映射")

                        output = self.model(x)
                        loss = self.l1_loss(output.squeeze(), y.float().detach())
                        # loss = self.loss_m4(output.squeeze(),x.size(0),y.float().detach())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
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
                self.have_epoch = self.have_epoch + 1
                # 计算每个 epoch 的平均 loss，并添加到 epoch_loss_history 中
                self.avg_epoch_loss = epoch_loss / self.max_iterations_per_round
                self.loss_history.append(self.avg_epoch_loss)
                Epoch_step = f'- Local_Epoch:{step} ' if max_local_epochs!=1 else ''
                logger.info(
                    f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id}{Epoch_step} - Average Loss: {self.avg_epoch_loss:.4f}")



            # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time



        else:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
            self.model.eval()
            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                # Forward pass
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                # Negative log likelihood as our loss
                loss = F.mse_loss(outputs, y)

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(loss, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

            # Evaluate on the client's test dataset
            # test_acc = self.evaluate()
            # print(f"Client {self.id}, Test Accuracy: {test_acc:.1f}, FIM-T value: {fim_trace_sum.item():.1f}")
            # print(f"FIM-T value change: {(self.fim_trace_history[-1] - (self.fim_trace_history[-2] if len(self.fim_trace_history) > 1 else 0)):.1f}")

    def load_train_data(self, batch_size=None):
        # 重写
        if batch_size == None:
            batch_size = self.batch_size
        if self.test_dataset == '':
            train_data = read_IQA_data(self.dataset, self.id, self.seed, is_train=True,
                                       selected_data_list=self.data_list,
                                       patch_size=self.patch_size, patch_num=self.patch_num)
        else:
            train_data = read_IQA_data_All(self.dataset, self.id, self.seed, is_train=True,
                                       selected_data_list=self.data_list,
                                       patch_size=self.patch_size, patch_num=self.patch_num,test_dataset=self.test_dataset)


        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True, num_workers=16)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # test_data = read_IQA_data(self.dataset, self.id, self.seed, is_train=False, selected_data_list=self.data_list,
        #                           patch_size=self.patch_size, patch_num=self.patch_num)

        if self.test_dataset == '':
            test_data = read_IQA_data(self.dataset, self.id, self.seed, is_train=False,
                                       selected_data_list=self.data_list,
                                       patch_size=self.patch_size, patch_num=self.patch_num)
        else:
            print("测试数据为"+self.test_dataset)
            test_data = read_IQA_data_All(self.dataset, self.id, self.seed, is_train=False,
                                       selected_data_list=self.data_list,
                                       patch_size=self.patch_size, patch_num=self.patch_num,test_dataset=self.test_dataset)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False, num_workers=16)

    def set_parameters(self, model, pfl_flag):
        # 只替换特征提取层
        if (pfl_flag):
            for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()

        else:  # 如果不是 PFL，则正常替换
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
            # print("整个模型替换")

    # 重写
    def test_metrics(self):
        # 重写test_metrics
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)

        self.model.eval()
        pred_scores = []
        gt_scores = []
        test_num = 0

        with torch.no_grad():
            steps2 = 0
            iteration = 0
            for img, label in tqdm(testloaderfull,
                                   desc=f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id} - Local  testing progress"):
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                # label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
                if (self.normalization):
                    label = torch.as_tensor(self.remap_mos(label.to(self.device),self.test_dataset)).requires_grad_(False)
                else:
                    label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
                pred = self.model(img)
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
        self.pre_SRCC = test_srcc
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
    #将mos值进行重映射
    def remap_mos(self,y,dataset_name):
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

    def loss_m4(self,y_pred_all,num,y_all):
        """prediction monotonicity related loss"""
        esp = 1e-8
        loss = 0

        # for task_num in per_num:  # per_num  每个任务包含的样本数，表示不同任务的数据量。
        y_pred = y_pred_all[:num]
        y = y_all[:num]

        # assert y_pred.size(0) > 1  #
        if y_pred.size(0) == 0:
            return 0
        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()  # 这是一个矩阵，其中的每个元素表示两个样本之间的预测值差异。
        gts = y - y.t()  # 这是一个矩阵，其中的每个元素表示两个样本之间的真实值差异。

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss += torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

        # loss = loss / len(per_num)

        return loss