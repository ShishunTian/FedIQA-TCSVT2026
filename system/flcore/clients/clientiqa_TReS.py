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
from utils.data_utils import read_IQA_data
from utils.privacy import *
import math

# from torchviz import make_dot
import logging

logger = logging.getLogger()


class clientIQA_TReS(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []  # 保存费舍尔信息矩阵的历史记录，用于客户端参数个性化时的加权操作。
        self.l1_loss = torch.nn.L1Loss()
        # self.loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=args.learning_rate_decay_gamma)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
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

    def train(self, is_selected):
        if is_selected:
            """
                    执行本地训练，确保每次只进行指定数量的迭代（iteration），而不是训练整个 epoch。               
            """
            self.Communication_num += 1
            trainloader = self.load_train_data()
            # 创建迭代器
            if not hasattr(self,
                           'train_iter') or self.train_iter is None or self.Communication_num % self.data_spilt_factor == 1:
                # 如果迭代器不存在，或者已经耗尽，创建新的迭代器
                self.train_iter = iter(trainloader)
            self.model.train()

            # 保存 loss 的历史记录
            self.avg_epoch_loss = []

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

                        self.model.zero_grad()

                        # [Fix Start] 修改了解包逻辑，防止返回过多值导致报错
                        output = self.model(x)
                        if isinstance(output, (list, tuple)):
                            pred = output[0]
                            closs = output[1]
                        else:
                            # 如果只返回了一个值（例如使用了DataParallel但没处理好，或者模型定义有误）
                            pred = output
                            closs = torch.tensor(0.0).to(self.device)

                        # 对翻转输入的同样处理
                        output_flip = self.model(torch.flip(x, [3]))
                        if isinstance(output_flip, (list, tuple)):
                            pred2 = output_flip[0]
                            closs2 = output_flip[1]
                        else:
                            pred2 = output_flip
                            closs2 = torch.tensor(0.0).to(self.device)
                        # [Fix End]

                        loss_qa = self.l1_loss(pred.squeeze(), y.float().detach())
                        loss_qa2 = self.l1_loss(pred2.squeeze(), y.float().detach())
                        # =============================================================================
                        # =============================================================================
                        # # 三元组损失计算
                        indexlabel = torch.argsort(y)  # small--> large
                        anchor1 = torch.unsqueeze(pred[indexlabel[0], ...].contiguous(), dim=0)  # d_min
                        positive1 = torch.unsqueeze(pred[indexlabel[1], ...].contiguous(), dim=0)  # d'_min+
                        negative1_1 = torch.unsqueeze(pred[indexlabel[-1], ...].contiguous(), dim=0)  # d_max+

                        anchor2 = torch.unsqueeze(pred[indexlabel[-1], ...].contiguous(), dim=0)  # d_max
                        positive2 = torch.unsqueeze(pred[indexlabel[-2], ...].contiguous(), dim=0)  # d'_max+
                        negative2_1 = torch.unsqueeze(pred[indexlabel[0], ...].contiguous(), dim=0)  # d_min+

                        # =============================================================================
                        # =============================================================================

                        fanchor1 = torch.unsqueeze(pred2[indexlabel[0], ...].contiguous(), dim=0)
                        fpositive1 = torch.unsqueeze(pred2[indexlabel[1], ...].contiguous(), dim=0)
                        fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1], ...].contiguous(), dim=0)

                        fanchor2 = torch.unsqueeze(pred2[indexlabel[-1], ...].contiguous(), dim=0)
                        fpositive2 = torch.unsqueeze(pred2[indexlabel[-2], ...].contiguous(), dim=0)
                        fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0], ...].contiguous(), dim=0)

                        consistency = nn.L1Loss()
                        assert (y[indexlabel[-1]] - y[indexlabel[1]]) >= 0
                        assert (y[indexlabel[-2]] - y[indexlabel[0]]) >= 0
                        triplet_loss1 = nn.TripletMarginLoss(margin=(y[indexlabel[-1]] - y[indexlabel[1]]),
                                                             p=1)  # d_min,d'_min,d_max
                        # triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
                        triplet_loss2 = nn.TripletMarginLoss(margin=(y[indexlabel[-2]] - y[indexlabel[0]]), p=1)
                        # triplet_loss1 = nn.TripletMarginLoss(margin=label[indexlabel[-1]], p=1)
                        # triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
                        tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
                                        triplet_loss2(anchor2, positive2, negative2_1)
                        ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
                                         triplet_loss2(fanchor2, fpositive2, fnegative2_1)
                        loss = loss_qa + closs + loss_qa2 + closs2 + 0.5 * (
                                self.l1_loss(tripletlosses, ftripletlosses.detach())
                                + self.l1_loss(ftripletlosses, tripletlosses.detach())) + 0.05 * (
                                           tripletlosses + ftripletlosses)
                        # loss = self.l1_loss(output.squeeze(), y.float().detach())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        # 累加当前batch的loss到epoch_loss中
                        epoch_loss += loss.item()
                        self.sum_loss += loss.item() * y.shape[0]
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
                Epoch_step = f'- Local_Epoch:{step} ' if max_local_epochs != 1 else ''
                logger.info(
                    f"({self.data_list[self.id].center(self.max_name_length)})ID: {self.id}{Epoch_step} - Average Loss: {self.avg_epoch_loss:.4f}")

            # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

        # [代码后续部分保持不变...]

    def load_train_data(self, batch_size=None):
        # 重写
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_IQA_data(self.dataset, self.id, self.seed, is_train=True, selected_data_list=self.data_list,
                                   patch_size=self.patch_size, patch_num=self.patch_num)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True, num_workers=16)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_IQA_data(self.dataset, self.id, self.seed, is_train=False, selected_data_list=self.data_list,
                                  patch_size=self.patch_size, patch_num=self.patch_num)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False, num_workers=16)

    def set_parameters(self, model):
        # 只替换特征提取层
        # for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
        #     old_param.data = new_param.data.clone()
        # 替换除了 fc2 和 fc 层之外的其他参数
        for (new_name, new_param), (old_name, old_param) in zip(model.named_parameters(),
                                                                self.model.named_parameters()):
            # 跳过 fc2 和 fc 层
            if not (("fc2" in old_name) or ("fc" in old_name)):
                old_param.data = new_param.data.clone()

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
                # if iteration<50:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                # [Fix Start] test_metrics 中也需要同样的保护措施
                output = self.model(img)
                if isinstance(output, (list, tuple)):
                    pred = output[0]
                else:
                    pred = output
                # [Fix End]

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

                # [Fix Start] train_metrics 中也需要同样的保护措施
                output = self.model(img)
                if isinstance(output, (list, tuple)):
                    pred = output[0]
                else:
                    pred = output
                # [Fix End]

                loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
                losses += loss_qa * label.shape[0]
                train_num += label.shape[0]
                iteration = iteration + 1
                # else:break

        return train_num, losses