import h5py
import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import logging

logger = logging.getLogger()


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))

def create_mean_table(mean_values, variance_values, data_list, folder_path):
    # 提取均值
    srcc_values = [mean_values.get(f'{data}_SRCC') for data in data_list]
    plcc_values = [mean_values.get(f'{data}_PLCC') for data in data_list]

    # 提取其他均值
    mean_srcc = mean_values.get('mean_srcc', None)
    mean_plcc = mean_values.get('mean_plcc', None)
    weight_srcc = mean_values.get('weight_srcc', None)
    weight_plcc = mean_values.get('weight_plcc', None)

    # 提取标准差
    var_srcc_values = [round(variance_values.get(f'{data}_SRCC_std'),3) for data in data_list]
    var_plcc_values = [round(variance_values.get(f'{data}_PLCC_std'),3) for data in data_list]

    # 创建 DataFrame
    mean_table = pd.DataFrame({
        'Client ID': range(len(data_list)),
        'Dataset': data_list,
        'SRCC': srcc_values,
        'PLCC': plcc_values,
        'SRCC Variance': var_srcc_values,
        'PLCC Variance': var_plcc_values,
    })

    # 添加均值的行
    mean_table = mean_table.append({
        'Client ID': 'Mean',
        'Dataset': '',
        'SRCC': mean_srcc,
        'PLCC': mean_plcc,
        'SRCC Variance': '',
        'PLCC Variance': ''
    }, ignore_index=True)

    # 添加权重的行
    mean_table = mean_table.append({
        'Client ID': 'Weight',
        'Dataset': '',
        'SRCC': weight_srcc,
        'PLCC': weight_plcc,
        'SRCC Variance': '',
        'PLCC Variance': ''

    }, ignore_index=True)

    # 使用 tabulate 打印结果
    logging.info("\n" + tabulate(mean_table, headers='keys', tablefmt='pretty', showindex=False))

    # 保存为 CSV 文件
    csv_file_path = os.path.join(folder_path, 'mean_values.csv')
    mean_table.to_csv(csv_file_path, index=False)
    logging.info(f"均值和方差结果已保存到 {csv_file_path}")

    return mean_table

def median_data(args, folder_path="", times=10):
    folders = [folder_path + f'/run_{i}' for i in range(times)]
    file_name = 'evaluation_results.csv'

    # 用于保存每个文件夹中最后一个为 TRUE 的行号
    last_true_indices = []
    # 用于保存每个文件夹中最后一个为 TRUE 的行的其他值（字典形式）
    last_true_values = []

    # 遍历每个文件夹
    for folder in folders:
        file_path = os.path.join(folder, file_name)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 找到 'mean_Updated' 列中最后一个为 TRUE 的行号
        true_indices = df.index[df['mean_Updated'] == True].tolist()

        if true_indices:
            last_true_index = true_indices[-1]  # 获取最后一个 TRUE 的行号
            last_true_indices.append(last_true_index)  # 保存行号

            # 获取该行的其他值（字典形式）
            other_values = df.loc[last_true_index].to_dict()
            last_true_values.append(other_values)  # 保存该行的其他值

    # 动态生成要计算均值和方差的列名
    columns_to_calculate = []
    for i in range(args.num_clients):
        columns_to_calculate.append(f'{args.select_data_list[i]}_SRCC')
        columns_to_calculate.append(f'{args.select_data_list[i]}_PLCC')

    # 额外的列
    columns_to_calculate.extend(['mean_srcc', 'mean_plcc', 'weight_srcc', 'weight_plcc'])

    # 初始化一个字典用于存储每个列的均值和方差
    mean_values = {}
    std_values = {}

    # 计算每个列的均值和方差
    for column in columns_to_calculate:
        # 提取每个字典中该列的值，并将其转换为浮点数
        column_values = [float(d[column]) for d in last_true_values if column in d]

        if column_values:
            mean_values[column] = np.mean(column_values)
            std_values[f'{column}_std'] = np.std(column_values)  # 计算方差
        else:
            mean_values[column] = None
            std_values[f'{column}_std'] = None
            logging.info(f"\n未找到 '{column}' 的有效值")

    data_list = args.select_data_list[:args.num_clients]  # 选择前 num_clients 个数据集
    create_mean_table(mean_values, std_values, data_list, folder_path)

    return last_true_indices, last_true_values, mean_values  # 返回行号列表, 字典值, 和每列的均值

def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc

def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc