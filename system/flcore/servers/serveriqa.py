import copy
import csv
import os
import random
import time
import torch
import numpy as np
import h5py
# from flcore.clients.clientavg import clientAVG
from flcore.clients.clientiqa import clientIQA
from flcore.servers.serverbase import Server
from tabulate import tabulate
from utils.data_utils import read_IQA_data
import logging

logger = logging.getLogger()


class Fed_IQA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.patch_size = args.patch_size
        self.patch_num = args.patch_num
        self.seed = args.seed
        self.data_list = args.select_data_list
        self.best_weight_quality_metrics = {'SRCC': [-1], 'PLCC': [-1]}             ##保存按数据量加权的最好结果
        self.best_mean_quality_metrics = {'SRCC': [-1], 'PLCC': [-1]}               ##保存平均加权最好结果
        self.rs_mean_quality_metrics = {'SRCC': [-1], 'PLCC': [-1]}                 ##保存按平均加权的结果
        self.rs_weight_quality_metrics = {'SRCC': [-1], 'PLCC': [-1]}               ##保存按数据量加权的结果
        self.rs_raw_quality_metrics = {'SRCC': [-1], 'PLCC': [-1]}                  ##保存原始数据（各个数据集的）
        self.result_folder = args.result_folder
        self.tot_train_samples = 0
        self.tot_test_samples = 0
        self.pfl_flag = args.pfl

        self.set_clients(clientIQA)  # 初始化客户端，客户端使用 clientAS 类=

        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    # 重写set_clients
    def set_clients(self, clientObj):
        # 修改点：不再使用 zip 遍历，避免因 slow_clients 为空导致循环不执行
        for i in range(self.num_clients):
            # 安全获取 slow 状态，如果列表为空或索引越界，默认为 False
            train_slow = self.train_slow_clients[i] if len(self.train_slow_clients) > i else False
            send_slow = self.send_slow_clients[i] if len(self.send_slow_clients) > i else False

            train_data = read_IQA_data(self.dataset, i, seed=self.seed, is_train=True, selected_data_list=[],
                                       patch_size=self.patch_size, patch_num=self.patch_num)
            test_data = read_IQA_data(self.dataset, i, seed=self.seed, is_train=False, selected_data_list=[],
                                      patch_size=self.patch_size, patch_num=self.patch_num)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)
            self.tot_train_samples += len(train_data)
            self.tot_test_samples += len(test_data)

    # 重写test_metrics
    def test_metrics(self):
        # if self.eval_new_clients and self.num_new_clients > 0:
        #     self.fine_tuning_new_clients()
        #     return self.test_metrics_new_clients()
        tot_plcc = []
        tot_srcc = []
        weight_srcc = []#与数据量进行加权
        weight_plcc = []#与数据量进行加权
        for c in self.clients:
            test_num, srcc, plcc = c.test_metrics()
            tot_srcc.append(srcc)
            tot_plcc.append(plcc)
            weight_srcc.append(srcc*c.test_samples)
            weight_plcc.append(plcc * c.test_samples)
        ids = [c.id for c in self.clients]
        tot_weight_srcc = sum(weight_srcc)/self.tot_test_samples
        tot_weight_plcc = sum(weight_plcc)/self.tot_test_samples
        return ids, tot_plcc,tot_srcc,tot_weight_srcc,tot_weight_plcc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            ns = c.train_num
            cl = c.sum_loss
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    # 重写
    def evaluate(self,epoch,mean_quality_metrics=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        mean_plcc = np.mean(stats[1])
        mean_srcc = np.mean(stats[2])
        std_plcc = np.std(stats[1])
        std_srcc = np.std(stats[2])

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        # 保存原始数据
        self.rs_raw_quality_metrics['SRCC'].append(stats[2])  # 追加原始 SRCC 数据
        self.rs_raw_quality_metrics['PLCC'].append(stats[1])  # 追加原始 PLCC 数据

        if mean_quality_metrics is None:
            self.rs_mean_quality_metrics['SRCC'].append(mean_srcc)
            self.rs_mean_quality_metrics['PLCC'].append(mean_plcc)
        else:
            mean_quality_metrics['SRCC'].append(mean_srcc)
            mean_quality_metrics['PLCC'].append(mean_plcc)
        self.rs_weight_quality_metrics['SRCC'].append(stats[3])
        self.rs_weight_quality_metrics['PLCC'].append(stats[4])

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)


        #如果srcc大于保存值，输出Update the optimal results,并更新变量
        is_weight_updated = stats[3]>self.best_weight_quality_metrics['SRCC']
        is_mean_updated = mean_srcc > self.best_mean_quality_metrics['SRCC']
        if is_mean_updated:
            self.best_mean_quality_metrics['SRCC'] = mean_srcc
            self.best_mean_quality_metrics['PLCC'] = mean_plcc
            for c in self.clients:
                c.save_item(c.model.state_dict(), c.data_list[c.id], c.result_folder)
            logger.info("Update the  mean_optimal  results")
        if is_weight_updated:
            self.best_weight_quality_metrics['SRCC']=stats[3]
            self.best_weight_quality_metrics['PLCC']=stats[4]
            for c in self.clients:
                c.save_item(c.model.state_dict(), c.data_list[c.id] + "_weight", c.result_folder)
            logger.info("Update the weight_optimal results")


        table = []
        srcc_data = []
        plcc_data = []
        for i in range(len(self.alled_clients)):
            srcc_data.append(stats[2][i])  # SRCC for each client
            plcc_data.append(stats[1][i])  # PLCC for each client
            row = [i, self.data_list[i], stats[2][i], stats[1][i]]  # 每行的数据：[Client ID, SRCC, PLCC]
            table.append(row)
        # 设置表头
        headers = ["Client ID", "Dataset", "SRCC", "PLCC"]
        # 输出全局表头和表格
        logger.info("Global model test")
        logger.info(tabulate(table, headers, tablefmt="pretty"))

        #csv输出
        self.write_evaluation_results_to_csv(
            epoch=epoch,
            srcc_data=srcc_data,
            plcc_data=plcc_data,
            mean_srcc=mean_srcc,
            mean_plcc=mean_plcc,
            weight_srcc=stats[3],
            weight_plcc=stats[4],
            std_srcc=std_srcc,
            std_plcc=std_plcc,
            train_loss=train_loss,
            result_folder=self.result_folder,
            alled_clients=self.alled_clients,
            data_list=self.data_list,
            is_mean_updated=is_mean_updated,
            is_weight_updated=is_weight_updated
        )

        # 输出平均值和标准差
        logger.info("Averaged Test SRCC: {:.4f}, Averaged Test PLCC: {:.4f}".format(mean_srcc, mean_plcc))
        logger.info("  Std    Test SRCC: {:.4f},    Std   Test PLCC: {:.4f}".format(std_srcc, std_plcc))
        logger.info("Averaged Train Loss: {:.4f}".format(train_loss))

    def all_clients(self):
        return self.clients  # 返回所有客户端，供后续训练过程中选择。

    def send_selected_models(self, selected_ids):
        # 遍历所有客户端并过滤出选定的客户端，向客户端发送全局模型参数
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            # progress = epoch / self.global_rounds
            # 向客户端发送全局模型参数，同时计算发送和接收的时间开销。
            client.set_parameters(self.global_model,self.pfl_flag)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def train(self):
        for i in range(self.global_rounds):
            self.communication_num = i
            logger.info(f"\n-----------------------Round number: {i}-----------------------")
            s_t = time.time()
            self.selected_clients = self.select_clients()  # 选择参与的客户端 (self.select_clients())
            self.alled_clients = self.all_clients()  # 获取所有客户端 (self.all_clients()）。

            selected_ids = [client.id for client in self.selected_clients]

            # self.send_models()
            self.send_selected_models(selected_ids)  # 向选定客户端发送模型参数

            # # 不微调进行测试结果
            # if i>0:
            #     # logger.info(f"\n-----------------------Round number: {i}-----------------------")
            #     self.evaluate(i)  # 评估全局模型。

            #客户端训练
            for client in self.alled_clients:
                # print("===============")
                client.train(client.id in selected_ids)  # 被选中的客户端进行训练
            # 微调进行测试的结果
            if i % self.eval_gap == 0:
            # if i == 99:
                self.evaluate(i)  # 评估模型。

            # #保存模型，留一法用的
            # for c in self.clients:
            #     c.save_item(c.model.state_dict(), c.data_list[c.id] + "_weight", c.result_folder)


            # self.receive_models_dynamic()  # 接收客户端上传的模型参数
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info(f"time cost：{self.Budget[-1]}")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        # sort active_clients by client.id
        active_clients = sorted(active_clients, key=lambda client: client.id)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        tot_Srcc=0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            self.uploaded_ids.append(client.id)
            # # 按数据量加权
            # tot_samples += client.train_samples
            # self.uploaded_weights.append(client.train_samples)
            # 按srcc加权
            tot_Srcc += client.pre_SRCC
            self.uploaded_weights.append(client.pre_SRCC)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            # # 按数据量加权
            # self.uploaded_weights[i] = w / tot_samples
            # 按srcc加权
            # avg_Srcc = tot_Srcc/self.num_clients;
            # self.uploaded_weights[i] = w-avg_Srcc / tot_Srcc-avg_Srcc*self.num_clients
            # 平均加权
            self.uploaded_weights[i] = 1 / self.num_clients
