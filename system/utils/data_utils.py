import numpy as np
import os
import torch
import random

import torchvision
from utils import folders


#读取指定客户端 (idx) 的数据文件。该文件使用 .npz 格式存储，其中包含客户端的训练或测试数据。
def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('/data/zyh26/PFL', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('/data/zyh26/PFL', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def read_IQA_data(dataset, idx, seed=0,is_train=True,selected_data_list=[],patch_size=None,patch_num=None):
    random.seed(seed)
    DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
    # selected_data_list: 如果不为空，将使用此列表中的域加载数据集；否则，使用默认的 DATA_LIST
    using_list = DATA_LIST if selected_data_list == [] else selected_data_list

    folder_path = {
        'live': os.path.join('/data/zyh26/PFL', dataset,'LIVE/'),
        'csiq': os.path.join('/data/zyh26/PFL', dataset,'CSIQ/'),
        'tid2013': os.path.join('/data/zyh26/PFL', dataset,'TID2013/'),
        'kadid10k': os.path.join('/data/zyh26/PFL', dataset,'kadid10k'),
        'clive': os.path.join('/data/zyh26/PFL', dataset,'ChallengeDB_release'),
        'koniq': os.path.join('/data/zyh26/PFL', dataset,'koniq-10k'),
        # 'fblive': os.path.join('/mnt/mdisk/zyh26/PFL', dataset,''),
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid10k': list(range(0, 80)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        # 'fblive': list(range(0, 39810)),
    }
    total_num_images = img_num[using_list[idx]]
    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    #transforms
    if (using_list[idx] == 'live') | (using_list[idx] == 'csiq') | (using_list[idx] == 'tid2013') | (using_list[idx] == 'clive') | (
            using_list[idx] == 'kadid10k'):
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    elif using_list[idx] == 'koniq':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
    elif using_list[idx] == 'fblive':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])

    path = folder_path[using_list[idx]]
    if is_train:
        img_indx=train_index
        # train_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                       train_index, patch_size,
        #                                       patch_num,
        #                                       batch_size=batch_size, istrain=True)
        # return train_loader.get_data()
    else:
        img_indx = test_index
        # test_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                           test_index, patch_size,
        #                                           patch_num,
        #                                           batch_size=batch_size, istrain=False)

    if using_list[idx] == 'live':
        data = folders.LIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'clive':
        data = folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'csiq':
        data = folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'koniq':
        data = folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'fblive':
        data = folders.FBLIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'tid2013':
        data = folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'kadid10k':
        data = folders.Kadid10k(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    return data

#留一法
def read_IQA_data_All(dataset, idx, seed=0,is_train=True,selected_data_list=[],patch_size=None,patch_num=None,test_dataset=None):
    random.seed(seed)
    DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
    # selected_data_list: 如果不为空，将使用此列表中的域加载数据集；否则，使用默认的 DATA_LIST
    using_list = DATA_LIST if selected_data_list == [] else selected_data_list

    folder_path = {
        'live': os.path.join('/data/zyh26/PFL', dataset,'LIVE/'),
        'csiq': os.path.join('/data/zyh26/PFL', dataset,'CSIQ/'),
        'tid2013': os.path.join('/data/zyh26/PFL', dataset,'TID2013/'),
        'kadid10k': os.path.join('/data/zyh26/PFL', dataset,'kadid10k'),
        'clive': os.path.join('/data/zyh26/PFL', dataset,'ChallengeDB_release'),
        'koniq': os.path.join('/data/zyh26/PFL', dataset,'koniq-10k'),
        # 'fblive': os.path.join('/mnt/mdisk/zyh26/PFL', dataset,''),
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid10k': list(range(0, 80)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        # 'fblive': list(range(0, 39810)),
    }

    target_dataset = using_list[idx] if is_train else test_dataset

    if not target_dataset or target_dataset not in folder_path:
        raise ValueError("Invalid test_dataset specified or not in folder_path.")
    total_num_images = img_num[target_dataset]

    # print("target_dataset" + target_dataset)


    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    if is_train:
        img_indx = total_num_images[0:len(total_num_images)]
    else:
        img_indx = total_num_images[0:len(total_num_images)]  # 使用测试数据集的完整索引

    #transforms
    if (target_dataset == 'live') | (target_dataset == 'csiq') | (target_dataset == 'tid2013') | (target_dataset == 'clive') | (
            target_dataset == 'kadid10k'):
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    elif target_dataset == 'koniq':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
    elif target_dataset == 'fblive':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])

    path = folder_path[target_dataset]


    if target_dataset == 'live':
        data = folders.LIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'clive':
        data = folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'csiq':
        data = folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'koniq':
        data = folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'fblive':
        data = folders.FBLIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'tid2013':
        data = folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif target_dataset == 'kadid10k':
        data = folders.Kadid10k(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    return data



def read_IQA_data_Hyper(dataset, idx, seed=0,is_train=True,selected_data_list=[],patch_size=None,patch_num=None):
    random.seed(seed)
    DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
    # selected_data_list: 如果不为空，将使用此列表中的域加载数据集；否则，使用默认的 DATA_LIST
    using_list = DATA_LIST if selected_data_list == [] else selected_data_list

    folder_path = {
        'live': os.path.join('/data/zyh26/PFL', dataset,'LIVE/'),
        'csiq': os.path.join('/data/zyh26/PFL', dataset,'CSIQ/'),
        'tid2013': os.path.join('/data/zyh26/PFL', dataset,'TID2013/'),
        'kadid10k': os.path.join('/data/zyh26/PFL', dataset,'kadid10k'),
        'clive': os.path.join('/data/zyh26/PFL', dataset,'ChallengeDB_release'),
        'koniq': os.path.join('/data/zyh26/PFL', dataset,'koniq-10k'),
        # 'fblive': os.path.join('/mnt/mdisk/zyh26/PFL', dataset,''),
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid10k': list(range(0, 80)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        # 'fblive': list(range(0, 39810)),
    }
    total_num_images = img_num[using_list[idx]]
    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    #transforms
    if (using_list[idx] == 'live') | (using_list[idx] == 'csiq') | (using_list[idx] == 'tid2013') | (using_list[idx] == 'clive') | (
            using_list[idx] == 'kadid10k'):
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    elif using_list[idx] == 'koniq':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
    elif using_list[idx] == 'fblive':
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])

    path = folder_path[using_list[idx]]
    if is_train:
        img_indx=train_index
        # train_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                       train_index, patch_size,
        #                                       patch_num,
        #                                       batch_size=batch_size, istrain=True)
        # return train_loader.get_data()
    else:
        img_indx = test_index
        # test_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                           test_index, patch_size,
        #                                           patch_num,
        #                                           batch_size=batch_size, istrain=False)

    if using_list[idx] == 'live':
        data = folders.LIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'clive':
        data = folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'csiq':
        data = folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'koniq':
        data = folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'fblive':
        data = folders.FBLIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'tid2013':
        data = folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'kadid10k':
        data = folders.Kadid10k(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    return data

def read_IQA_data_DBCNN(dataset, idx, seed=0,is_train=True,selected_data_list=[],patch_size=None,patch_num=None):
    random.seed(seed)
    DATA_LIST = ['live', 'csiq', 'tid2013', 'kadid10k', 'clive', 'koniq']
    # selected_data_list: 如果不为空，将使用此列表中的域加载数据集；否则，使用默认的 DATA_LIST
    using_list = DATA_LIST if selected_data_list == [] else selected_data_list

    folder_path = {
        'live': os.path.join('/data/zyh26/PFL', dataset,'LIVE/'),
        'csiq': os.path.join('/data/zyh26/PFL', dataset,'CSIQ/'),
        'tid2013': os.path.join('/data/zyh26/PFL', dataset,'TID2013/'),
        'kadid10k': os.path.join('/data/zyh26/PFL', dataset,'kadid10k'),
        'clive': os.path.join('/data/zyh26/PFL', dataset,'ChallengeDB_release'),
        'koniq': os.path.join('/data/zyh26/PFL', dataset,'koniq-10k'),
        # 'fblive': os.path.join('/mnt/mdisk/zyh26/PFL', dataset,''),
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid10k': list(range(0, 80)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        # 'fblive': list(range(0, 39810)),
    }
    total_num_images = img_num[using_list[idx]]
    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]
    #前4个数据集，使用原论文方式
    if (using_list[idx] == 'live') | (using_list[idx] == 'clive'):
        if is_train:
            if using_list[idx] == 'live':
                crop_size = 432
            else:
                crop_size = 448
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    elif (using_list[idx] == 'csiq') | (using_list[idx] == 'tid2013'):
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    #kad，kon使用Hyper方式
    elif (using_list[idx] == 'kadid10k'):
        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
    elif using_list[idx] == 'koniq':

        if is_train:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])



    path = folder_path[using_list[idx]]
    if is_train:
        img_indx=train_index
        # train_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                       train_index, patch_size,
        #                                       patch_num,
        #                                       batch_size=batch_size, istrain=True)
        # return train_loader.get_data()
    else:
        img_indx = test_index
        # test_loader = iqa_data_loader.DataLoader(using_list[idx], folder_path[using_list[idx]],
        #                                           test_index, patch_size,
        #                                           patch_num,
        #                                           batch_size=batch_size, istrain=False)

    if using_list[idx] == 'live':
        data = folders.LIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'clive':
        data = folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'csiq':
        data = folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'koniq':
        data = folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'fblive':
        data = folders.FBLIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'tid2013':
        data = folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    elif using_list[idx] == 'kadid10k':
        data = folders.Kadid10k(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    return data


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

