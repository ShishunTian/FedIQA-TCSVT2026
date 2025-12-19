#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveriqa import Fed_IQA
from flcore.servers.serveriqa_TReS import Fed_IQA_TReS
from flcore.servers.serveriqa_Hyper import Fed_IQA_Hyper
from utils.logging_config import setup_logger
from flcore.trainmodel.TReS.TReS import TReS_net
import random

# 深度学习模型的定义和实现
from flcore.trainmodel.models import *


from utils.result_utils import average_data,median_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def run(args):

    if args.seed == 0:
        pass
    else:
        print('we are using the seed = {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    time_list = []  # 用于记录每次运行的时间。
    reporter = MemReporter()  # 监控和报告内存使用情况。
    model_str = args.model
    # 获取当前时间戳并格式化为字符串（只包含月、日和小时）
    timestamp_str = time.strftime("%m%d%H")
    # 设定结果文件夹路径
    result_folder = args.result_folder
    # 生成文件夹名称
    decay_status = f'{args.learning_rate_decay_gamma:.0e}' if args.learning_rate_decay else "false"
    # 根据训练数据集设置 folder_name
    print(args.pfl)
    pfl_prefix = 'pfl' if args.pfl else 'fl'
    select_data = f'{args.select_data_list[0]}' if args.num_clients==1 else f'{args.num_clients}'
    folder_name = f'{pfl_prefix}_{args.test_dataset}_{select_data}_{args.algorithm}_{args.model}_{args.local_learning_rate}_{decay_status}_{timestamp_str}'
    # 创建文件夹路径
    folder_path = os.path.join(result_folder, folder_name)
    # 创建文件夹（如果不存在）
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 将 folder_path 赋值给 args.result_folder
    args.result_folder = folder_path
    # 设置日志文件名
    log_filename = os.path.join(folder_path, f'{folder_name}.log')
    logger = setup_logger(log_filename)

    ##打印
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.info("\ncuda is not avaiable.\n")
        args.device = "cpu"
    logger.info("=" * 50)
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Local batch size: {}".format(args.batch_size))
    logger.info("Local steps: {}".format(args.local_epochs))
    logger.info("Local learing rate: {}".format(args.local_learning_rate))
    logger.info("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        logger.info("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    logger.info("Total number of clients: {}".format(args.num_clients))
    logger.info("Clients join in each round: {}".format(args.join_ratio))
    logger.info("Clients randomly join: {}".format(args.random_join_ratio))
    logger.info("Running times: {}".format(args.times))
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("Backbone: {}".format(args.model))
    logger.info("Using device: {}".format(args.device))
    if args.device == "cuda":
        logger.info("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # if "uniformEpoch" in args.algorithm:
    logger.info("data_spilt_factor:{}".format(args.data_spilt_factor))
    logger.info("Use seed: {}".format(args.seed))
    logger.info("PFL: {}".format('True' if args.pfl else 'False'))
    logger.info("=" * 50)

    #打印结束


    for i in range(0, args.times):
        if args.times != 1:
            args.result_folder = os.path.join(folder_path, f"run_{i}")
            if not os.path.exists(args.result_folder):
                os.makedirs(args.result_folder)
        logger.info(f"\n========================== Running time: {i}th ==========================")
        logger.info("Creating server and clients ...")

        start = time.time()


        if model_str == "resnet":
            if "IQA" in args.dataset:
                args.model = torchvision.models.resnet50(pretrained=True).to(args.device)
                # 替换最后的全连接层用于回归任务
                # args.model.fc = torch.nn.Linear(args.model.fc.in_features, 1).to(args.device)
                args.model.fc = torch.nn.Sequential(
                    torch.nn.Linear(args.model.fc.in_features, 1024),  # 2048 -> 1024
                    torch.nn.ReLU(),  # 添加 ReLU 激活函数
                    torch.nn.Linear(1024, 1)  # 1024 -> 1
                ).to(args.device)
        elif model_str == "vgg16":
            if "IQA" in args.dataset:
                args.model = torchvision.models.vgg16(pretrained=True).to(args.device)
                # print(args.model)
                # 替换 VGG16 的最后分类层用于回归任务
                args.model.classifier[-1] = torch.nn.Sequential(
                    torch.nn.Linear(args.model.classifier[-1].in_features, 1024),  # 4096 -> 1024
                    torch.nn.ReLU(),  # 添加 ReLU 激活函数
                    torch.nn.Linear(1024, 1)  # 1024 -> 1
                ).to(args.device)
        elif model_str == "transformer":
            if "IQA" in args.dataset:
                args.model = torchvision.models.vit_b_16(pretrained=True).to(args.device)
                # 替换最后的全连接层用于回归任务
                args.model.heads = torch.nn.Sequential(
                    torch.nn.Linear(768, 512),  # 先减少维度
                    torch.nn.ReLU(),  # 添加 ReLU 激活函数
                    torch.nn.Linear(512, 1)  # 最终输出为 1
                ).to(args.device)
        elif model_str == "TReS":
            if "IQA" in args.dataset:
                args.model = TReS_net(device=args.device).to(args.device)
        elif model_str == "Hyper":
            pass
        else:
            raise NotImplementedError
        if i==0:
            logger.info(args.model)

        # select algorithm

        if args.algorithm == "FedIQA":
            if(model_str == "resnet"):
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = Fed_IQA(args, i)
            elif(model_str == "vgg16"):
                args.head = copy.deepcopy(args.model.classifier)
                args.model.classifier = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = Fed_IQA(args, i)
            elif(model_str == "transformer"):
                args.head = copy.deepcopy(args.model.heads)
                args.model.heads = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
                server = Fed_IQA(args, i)
        elif args.algorithm == "FedIQA_TReS":
            server = Fed_IQA_TReS(args, i)
        elif args.algorithm == "FedIQA_Hyper":
            args.head = HyperHead().to(args.device)
            base = HyperBackbone().to(args.device)
            args.model = BaseHeadSplit(base, args.head).to(args.device)
            server = Fed_IQA_Hyper(args, i)
        else:
            raise NotImplementedError
        # 训练联邦学习模型
        server.train()

        time_list.append(time.time() - start)

    logger.info(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    median_data(args,folder_path=folder_path,times=args.times,)

    logger.info("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # iqa
    parser.add_argument('-pfl', "--pfl",type=int, default=1)
    parser.add_argument('-normalization', "--normalization", type=int, default=0)
    parser.add_argument('--test_dataset', dest='test_dataset', type=str, default='',
                        help='Dataset to be used for testing')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--patch_num', dest='patch_num', type=int, default=50,
                        help='Number of sample patches from training image')
    parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
    parser.add_argument('--select_data_list', nargs='+',
                        default=['csiq', 'tid2013', 'kadid10k', 'clive', 'koniq'],
                        help='List of datasets to select from. Default is the full list.')
    parser.add_argument('-rf', "--result_folder", type=str, default="/data/zyh26/PFL_result")
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="2")
    parser.add_argument('-data', "--dataset", type=str, default="IQA")
    parser.add_argument('-m', "--model", type=str, default="resnet")  # 模型选择
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)  # batchsize
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=2e-5,  # 本地学习率
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)  # 是否启用学习率衰减。
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=5e-4)  # 学习率衰减系数
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)  # 全局训练轮次
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,  # 本地训练轮次
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedIQA_TReS")  # 算法
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,  # 每轮全局训练时参与的客户端比例。
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,  # 是否随机客户端在线比率
                        help="Random ratio of clients per round")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')  # 保存实验结果的文件夹名称。
    parser.add_argument('-nc', "--num_clients", type=int, default=5, # 客户端数量
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=10,  # 实验运行次数。
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,  # 评估间隔轮数。
                        help="Rounds gap for evaluation")
    parser.add_argument('-dsf', "--data_spilt_factor", type=int, default=10)  #  data_spilt_factor

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    run(args)

