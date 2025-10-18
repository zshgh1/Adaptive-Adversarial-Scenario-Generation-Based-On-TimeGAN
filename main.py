# -*- coding: UTF-8 -*-
# Local modules
import argparse
import logging
import pandas as pd
import os
import pickle
import random
import shutil


# 3rd-Party Modules
import numpy as np
import torch
import json
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter

# Self-Written Modules
from data.data_preprocess import data_preprocess
from metrics.metric_utils import (
    feature_prediction, one_step_ahead_prediction, reidentify_score
)

from models.timegan import TimeGAN
from models.utils import timegan_trainer, timegan_generator
from models.evaluate import TestWithgenerated_data, evaluate_trajectory,get_evaluate_metrics

from scipy.fftpack import fft
import math
import time
from collections import Counter


def main(args):
    ##############################################
    # Initialize output directories
    ##############################################

    ## Runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")

    ## Data directory
    data_path = os.path.abspath("./data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    ## Output directories
    args.model_path = os.path.abspath(f"./output/{args.exp}/")
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    print(f"\nCode directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    data_path = "data/smoothed_data.csv"
    #data_path = "data/merged_data_highD.csv"
    X, T, params, args.max_seq_len, args.padding_value = data_preprocess(
        data_path, args.max_seq_len
    )



    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
    print(f"Original data preview:\n{X[:2, :10, :2]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]

    # Train-Test Split data and time
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, test_size=args.test_rate, random_state=args.seed
    )
    #train_time = np.tile(train_time*2, 2)
    train_time = [time for time in train_time for _ in range(10)]
    train_time = np.array(train_time)
    # print(train_time)

    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()

    model = TimeGAN(args)
    # model.load_state_dict(torch.load("D:/pythonProject/pythonProject/timegan-pytorch/output/test_hidden8_e2000_q600_0.8/model_epoch_1800.pt"))
    if args.is_train == False:
        timegan_trainer(model, train_data, train_time, args)
    #generated_data = timegan_generator(model, train_time, args,f"{args.model_path}/model.pt")
    file = "model_epoch_1000.pt"

    if(args.max_seq_len==400):
        file="model_epoch_1500.pt"

    if (args.max_seq_len == 600):  
        file = "model_epoch_1500.pt"

    if (args.max_seq_len == 800):
        file = "model_epoch_1500.pt"

    if (args.max_seq_len == 200):
        file = "model_epoch_600.pt"

    generated_data = timegan_generator(model, train_time, args, f"{args.model_path}/{file}")
    print(generated_data.shape)
    generated_time = train_time
    reverse_generated_data = reverse_data(generated_data, params, data_path,scaling_method="standard")
    reverse_train_data = reverse_data(train_data, params,None, scaling_method="standard")
    reverse_test_data = reverse_data(test_data, params,None,  scaling_method="standard")
    # Log end time
    end = time.time()

    print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start)/60} mins\n")

    get_evaluate_metrics(reverse_train_data,args.scaling,args.platoonSize,args.mode_params,args.controllers)
    for ctrl in args.controllers:
        print(ctrl)
        generated_data_e=evaluate_trajectory(reverse_generated_data,args.scaling,
                                             ctrl,args.platoonSize,args.mode_params)
        # final_data = np.asarray([traj.m_vellist for traj in generated_data1])
        # np.savetxt(f"{args.model_path}/generated_data.txt", final_data, fmt='%0.3f')
        # final_data1 = np.asarray([traj.m_vellist for traj in train_data1])
        # np.savetxt(f"{args.model_path}/train_data.txt", final_data1, fmt='%0.3f')

        columns = ['Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway']
        generated_data_e_2d, generated_vehicle_e_ids = convert_3d_to_2d(generated_data_e)
        generated_df = pd.DataFrame(generated_data_e_2d, columns=columns[1:])
        generated_df.insert(0, 'Vehicle_ID', generated_vehicle_e_ids)

        generated_df.to_csv(f"{args.model_path}/fake_data_{ctrl}_P{args.platoonSize}_ttc.csv", index=False)




    # model_files = [f for f in os.listdir(args.model_path) if f.endswith('.pt')]
    # for model_file in model_files:
    #     model_path = os.path.join(args.model_path, model_file)
    #     generated_data = timegan_generator(model, train_time, args, model_path)
    #     # 处理生成数据
    #     reverse_generated_data = reverse_data(generated_data, params, data_path, scaling_method="standard")
    #
    #     # generated_data1 = evaluate_metrics(reverse_generated_data[:, :, 2])
    #     # final_data = np.asarray([traj.m_vellist for traj in generated_data1])
    #     # output_filename = f"fake_data_{model_file.split('.')[0]}.txt"
    #     # output_path = os.path.join(args.model_path, output_filename)
    #     # np.savetxt(output_path, final_data, fmt='%0.3f')
    #     # 转换为2D并保存
    #     columns = ['Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway']
    #     generated_data_2d, generated_vehicle_ids = convert_3d_to_2d(reverse_generated_data)
    #     generated_df = pd.DataFrame(generated_data_2d, columns=columns[1:])
    #     generated_df.insert(0, 'Vehicle_ID', generated_vehicle_ids)
    #     # 动态生成文件名（如 model_epoch_100.pt → fake_data_model_epoch_100.csv）
    #     output_filename = f"fake_data_{model_file.split('.')[0]}.csv"
    #     output_path = os.path.join(args.model_path, output_filename)
    #     generated_df.to_csv(output_path, index=False)
    #
    #     output_filename = f"fake_data_{model_file.split('.')[0]}.pickle"
    #     output_path = os.path.join(args.model_path, output_filename)
    #
    #     with open(output_path, "wb") as fb:
    #         pickle.dump(generated_data, fb)


    #########################
    # Save train and generated data for visualization
    #########################

    # # 设置 numpy 打印选项
    # np.set_printoptions(precision=5, suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    #
    # # 假设 train_data、test_data、generated_data、generated_time、test_time 已经定义
    #
    # # 定义表头
    # columns = ['Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Space_Headway', 'Time_Headway']
    # # 处理训练数据
    # train_data_2d, train_vehicle_ids = convert_3d_to_2d(reverse_train_data)
    # train_df = pd.DataFrame(train_data_2d, columns=columns[1:])
    # train_df.insert(0, 'Vehicle_ID', train_vehicle_ids)
    # train_df.to_csv(f"{args.model_path}/train_data.csv", index=False)
    # # 处理测试数据
    # test_data_2d, test_vehicle_ids = convert_3d_to_2d(reverse_test_data)
    # test_df = pd.DataFrame(test_data_2d, columns=columns[1:])
    # test_df.insert(0, 'Vehicle_ID', test_vehicle_ids)
    # test_df.to_csv(f"{args.model_path}/test_data.csv", index=False)
    # # 处理生成数据
    # generated_data_2d, generated_vehicle_ids = convert_3d_to_2d(reverse_generated_data)
    # generated_df = pd.DataFrame(generated_data_2d, columns=columns[1:])
    # generated_df.insert(0, 'Vehicle_ID', generated_vehicle_ids)
    # generated_df.to_csv(f"{args.model_path}/fake_data.csv", index=False)
    #
    #
    #
    #
    # with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
    #     pickle.dump(train_data, fb)
    # with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
    #     pickle.dump(train_time, fb)
    # with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
    #     pickle.dump(test_data, fb)
    # with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
    #     pickle.dump(test_time, fb)
    # with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
    #     pickle.dump(generated_data, fb)
    # with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
    #     pickle.dump(generated_time, fb)



    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0)

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # 1. Feature prediction
    feat_idx = np.random.permutation(train_data.shape[2])[:args.feat_pred_no]
    print("Running feature prediction using original data...")
    ori_feat_pred_perf = feature_prediction(
        (train_data, train_time),
        (test_data, test_time),
        feat_idx,
        args.max_seq_len
    )
    print("Running feature prediction using generated data...")
    new_feat_pred_perf = feature_prediction(
        (generated_data, generated_time),
        (test_data, test_time),
        feat_idx,
        args.max_seq_len
    )

    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    print('Feature prediction results:\n' +
          f'(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n' +
          f'(2) New: {str(np.round(new_feat_pred_perf, 4))}\n')

    # 2. One step ahead prediction
    print("Running one step ahead prediction using original data...")
    ori_step_ahead_pred_perf = one_step_ahead_prediction(
        (train_data, train_time),
        (test_data, test_time),
        args.max_seq_len
    )
    print("Running one step ahead prediction using generated data...")
    new_step_ahead_pred_perf = one_step_ahead_prediction(
        (generated_data, generated_time),
        (test_data, test_time),
        args.max_seq_len
    )

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print('One step ahead prediction results:\n' +
          f'(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n' +
          f'(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n')

    print(f"Total Runtime: {(time.time() - start)/60} mins\n")

    return None

def convert_3d_to_2d(data):
    num_samples, seq_length, num_features = data.shape
    data_2d = data.reshape(-1, num_features)
    vehicle_ids = np.repeat(np.arange(num_samples), seq_length)
    return data_2d, vehicle_ids
def reverse_data(generated_data, params, new_path,scaling_method="standard"):
    if scaling_method == "minmax":
        data_min = params[0][1:]  # 排除 ID 列
        data_max = params[1][1:]  # 排除 ID 列
        # MinMaxScaler 逆变换公式
        reversed_data = generated_data * (data_max - data_min) + data_min
    elif scaling_method == "standard":
        mean = params[0][1:]  # 排除 ID 列
        var = params[1][1:]  # 排除 ID 列
        std = np.sqrt(var)
        # StandardScaler 逆变换公式
        reversed_data = generated_data * std + mean

    if new_path:
        real_df = pd.read_csv(new_path)
        grouped = real_df.groupby('Vehicle_ID')
        real_initial_vels = [group['v_Vel'].iloc[0] for _, group in grouped]
        for i in range(reversed_data.shape[0]):
            velocity_sequence = reversed_data[i, :, 2].copy()
            if len(velocity_sequence) > 0 and not np.isnan(velocity_sequence[0]):
                # 原始初速度
                original_initial_vel = velocity_sequence[0]
                new_initial_vel = np.random.choice(real_initial_vels)
                velocity_offset = new_initial_vel - original_initial_vel
                window_length = 22  # 窗口长度
                poly_order = 3  # 多项式阶数
                velocity_sequence = savgol_filter(velocity_sequence, window_length, poly_order)

                reversed_data[i, :, 2] = velocity_sequence + velocity_offset
                reversed_data[i, :, 2] = np.maximum(0, reversed_data[i, :, 2])

    return reversed_data

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    mode_params = {
        "PID": "kp=1.54,ki=0.25,kd=0.23",
        "IDM": "a=4.0,b=4.0,delta=2.0,T=0.8",
        "CACC": "j1=0.25,j2=1.76,j3=0.26,veh_s0=2,veh_Tc=0.8",
        "CACC_TPF": "j1=0.25,j2=1.76,j3=0.26,veh_s0=2,veh_Tc=0.8,alpha=0.5",
        "LCM": "A=10.0,h_T=0.8,s0=2",
        "PID_safe": "kp=1.54,ki=0.25,kd=0.23,distance_error_com=0.05,"
                    "speed_front_error=0,acc_front_min=-5.0,acc_self_min=-5.0,time_perception=0.3,time_communicate=0.2,"
                    "time_wait_per=0.05,time_wait_com=0.05,period=0.1",
        "OVM": "a=0.1,b=1.75,vf=15",
        "FVD": "C1=0.13,C2=2.79,V1=16,V2=16,alpha=0.1,lambda=2.0,s0=7.0",
        "GM4": "alpha=2.5",
    }
    # "PID_safe": "kp=1.54,ki=0.25,kd=0.23,distance_error_com=0.8,"
    # "speed_front_error=-0.1,acc_front_min=-8.0,acc_self_min=-8.0,time_perception=0.3,time_communicate=0.2,"
    # "time_wait_per=0.05,time_wait_com=0.05,period=0.1",
    #CONTROLLERS = ["LCM", "CACC", "IDM", "PID", "PID_safe", "OVM", "GM4"]
    CONTROLLERS = ["PID_safe"]
    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test_hidden8_e2000_q800_0.8',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=800,
        type=int)
    parser.add_argument(
        '--test_rate',
        default=0.2,
        type=float)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=2000,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=8,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float)
    parser.add_argument(
        '--mode_params',default=mode_params,
        type=dict)
    parser.add_argument(
        '--scaling',
        default=0.3,
        type=float)
    parser.add_argument(
        '--platoonSize',
        default=2,
        type=int)
    parser.add_argument(
        '--controllers',
        default=CONTROLLERS,
        type=list)
    args = parser.parse_args()

    # Call main function
    main(args)

