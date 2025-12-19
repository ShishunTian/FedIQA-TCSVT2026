#'live' 'csiq' 'tid2013' 'kadid10k' 'clive' 'koniq'
#dsf 数据集分割
python main.py --seed 2021 -did 1 -data IQA --patch_num 50 -m Hyper -lbs 96 -dsf 10 -t 10 -gr 160 -ls 1 -algo FedIQA_uniformEpoch_Hyper -nc 5 --select_data_list 'csiq' 'tid2013' 'kadid10k' 'clive' 'koniq'
