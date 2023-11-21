import os
import pandas as pd
import torch
import random
import numpy as np
import argparse
from score import Score
## Define the subset of dataset class to use here:
Electricity_SubDataset = ["FOOD1"]

# seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='EasyTS: The Express Lane to Long Time Series Forecasting')

## Define the dataset class to use here:
parser.add_argument('--dataset_class', type=list, default=['Electricity'], help='Data class for evaluation')

## Define the model to be used here:
parser.add_argument('--model', type=str, default='DLinear', help='model name')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# Training settings
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# Forecasting settings
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task following by Informer')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# Optimization
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='The final score is the average of the number of experiments')
parser.add_argument('--training_metric', type=str, default='mse', help='metric during training')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.is_training:
    sign = 1
    # Reading Different Dataset Class
    for i in range(len(args.dataset_class)):
        current_dataset = args.dataset_class[i]
        subdataset = eval(current_dataset + '_SubDataset')
        # Reading Different Dataset Subsets
        for ii in subdataset:
            metrics_df = pd.DataFrame()
            for iii in range(args.itr):
                current_setting = '{}_on|{}-{}|_seq|{}-{}-{}|_freq|{}|_{}_{}'.format(
                    args.model,
                    current_dataset,
                    ii,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.freq,
                    args.target,
                    iii
                )

                score = Score(args, current_dataset, ii)
                print('####  start score on {} , progress {}/{}'.format(current_dataset, i, len(args.dataset_class)))

                print('####  start training : {}#########'.format(current_setting))
                score.train(current_setting)

                print('####  start testing : {}##########'.format(current_setting))
                metrics_dict = score.test(current_setting)
                temp_df = pd.DataFrame([metrics_dict])
                metrics_df = pd.concat([metrics_df, temp_df], axis=0)
                torch.cuda.empty_cache()
            # Calculate the average score for multiple iterations
            itr_means = pd.DataFrame(metrics_df.mean()).T
            itr_means.insert(0, 'dataset', ii)
            itr_means.insert(0, 'model', args.model)
            # Save result to CSV file (create if it doesn't exist; append data if it already exists)
            if sign:
                itr_means.to_csv('metrics.csv', mode='w',index=False)
                sign = 0
            else:
                itr_means.to_csv('metrics.csv', mode='a', header=False, index=False)
    print('####  task finish    ##########')