# logging_config.py

import logging
import os
import time


import logging

def setup_logger(log_filename):

    log_directory = './log/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 手动写入时间戳作为日志的第一行
    with open(log_filename, 'w') as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Log started at: {timestamp}\n")

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 或者你需要的其他级别

    # 创建一个文件处理器，使用'w'模式，重新开始日志
    file_handler = logging.FileHandler(log_filename, mode='a')  # 使用'a'以附加到已有的时间戳
    file_handler.setLevel(logging.INFO)  # 或者其他级别

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter('%(message)s'))  # 去掉时间戳

    # 创建格式化器
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)

    # 刷新日志输出
    file_handler.flush()

    return logger