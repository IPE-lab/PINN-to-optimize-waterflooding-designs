# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import mat73
import numpy as np
import json
from tqdm import tqdm 
from model.PINNModel import PINNModel
from model.PureMachineLearning import PureMachineLearning
from random import shuffle

import torch
import json
# well_adjecent_index = {0: [0], 1: [0, 1], 2: [1], 3: [
#     0, 2], 4: [0, 1, 2, 3], 5: [1, 3], 6: [2], 7: [2, 3], 8: [3]}
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INJECTRATE = mat73.loadmat('./data/TrainingData/Tranining_data/injrate.mat')['sample']
OPR = mat73.loadmat('./data/TrainingData/Tranining_data/OPR.mat')['OPR']
WPR = mat73.loadmat('./data/TrainingData/Tranining_data/WPR.mat')['WPR']
TIME = mat73.loadmat('./data/TrainingData/Tranining_data/Time.mat')['Time']

INJECTRATE_200 = mat73.loadmat('./data/200cases/injrate.mat')['sample']
OPR_200 = mat73.loadmat('./data/200cases/OPR.mat')['OPR']
WPR_200 = mat73.loadmat('./data/200cases/WPR.mat')['WPR']
TIME_200 = mat73.loadmat('./data/200cases/Time.mat')['Time']

INJECTRATE = np.concatenate((INJECTRATE, INJECTRATE_200), axis=0)
OPR = np.concatenate((OPR, OPR_200), axis=1)
WPR = np.concatenate((WPR, WPR_200), axis=1)

def read_training_data(case_ind):
    inject_rate = INJECTRATE[case_ind:case_ind+1, :]
    slice_index = np.arange(case_ind*13, (case_ind+1)*13)
    opr = OPR[:, slice_index][:, 4:]
    wpr = WPR[:, slice_index][:, 4:]
    time = TIME.reshape(-1, 1)
    return inject_rate, opr, wpr, time

def load_training_data(case_ind):
    injection_rate, opr, wpr, time = read_training_data(case_ind)
    BHP = 500
    train_producer_bottom_pressure = np.ones((opr.shape[0], 9)) * BHP
    train_water_inject_rate = np.ones((opr.shape[0], 4)) * injection_rate
    train_producer_bottom_pressure = torch.tensor(train_producer_bottom_pressure, dtype=torch.float, requires_grad=True)
    train_water_inject_rate = torch.tensor(train_water_inject_rate, dtype=torch.float, requires_grad=True)
    train_T = torch.tensor(time, dtype=torch.float, requires_grad=True)
    train_OPR = torch.tensor(opr, dtype=torch.float, requires_grad=True)
    train_WPR = torch.tensor(wpr, dtype=torch.float, requires_grad=True)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR

def load_all_training_data(case_list = list(range(45))):
    train_water_inject_rate = []
    train_producer_bottom_pressure = []
    train_T = []
    train_OPR = []
    train_WPR = []
    for i in case_list:
        temp_train_water_inject_rate, temp_train_producer_bottom_pressure, temp_train_T, temp_train_OPR, temp_train_WPR = load_training_data(i)
        train_water_inject_rate.append(temp_train_water_inject_rate)
        train_producer_bottom_pressure.append(temp_train_producer_bottom_pressure)
        train_T.append(temp_train_T)
        train_OPR.append(temp_train_OPR)
        train_WPR.append(temp_train_WPR)
    train_water_inject_rate = torch.cat(train_water_inject_rate, dim=0)
    train_producer_bottom_pressure = torch.cat(train_producer_bottom_pressure, dim=0)
    train_T = torch.cat(train_T, dim=0)
    train_OPR = torch.cat(train_OPR, dim=0)
    train_WPR = torch.cat(train_WPR, dim=0)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR


def read_test_data(case_name='case1'):
    # obtain data from simulator
    train_OPR = mat73.loadmat(f'./data/{case_name}/OPR.mat')['OPR'][:, 4:]
    train_WPR = mat73.loadmat(f'./data/{case_name}/WPR.mat')['WPR'][:, 4:]
    train_T = mat73.loadmat(f'./data/{case_name}/Time.mat')['Time'].reshape(-1, 1)

    BHP = 500
    with open(f'./data/{case_name}/params.json', 'r') as f:
        params = json.load(f)

    well_name = ['inject_rate_well_1', 'inject_rate_well_2',
                 'inject_rate_well_3', 'inject_rate_well_4']
    injection_rate = np.array([[params[key] for key in well_name]])

    train_producer_bottom_pressure = np.ones((train_OPR.shape[0], 9)) * BHP
    train_water_inject_rate = np.ones((train_OPR.shape[0], 4)) * injection_rate
    train_producer_bottom_pressure = torch.tensor(train_producer_bottom_pressure, dtype=torch.float, requires_grad=True)
    train_water_inject_rate = torch.tensor(train_water_inject_rate, dtype=torch.float, requires_grad=True)
    train_T = torch.tensor(train_T, dtype=torch.float, requires_grad=True)
    train_OPR = torch.tensor(train_OPR, dtype=torch.float, requires_grad=True)
    train_WPR = torch.tensor(train_WPR, dtype=torch.float, requires_grad=True)
    return train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR

def load_all_test_data():
    test_water_inject_rate = []
    test_producer_bottom_pressure = []
    test_T = []
    test_OPR = []
    test_WPR = []
    for i in range(5):
        case_name = f'case{i}'
        temp_test_water_inject_rate, temp_test_producer_bottom_pressure, temp_test_T, temp_test_OPR, temp_test_WPR = read_test_data(case_name)
        test_water_inject_rate.append(temp_test_water_inject_rate)
        test_producer_bottom_pressure.append(temp_test_producer_bottom_pressure)
        test_T.append(temp_test_T)
        test_OPR.append(temp_test_OPR)
        test_WPR.append(temp_test_WPR)
    test_water_inject_rate = torch.cat(test_water_inject_rate, dim=0)
    test_producer_bottom_pressure = torch.cat(test_producer_bottom_pressure, dim=0)
    test_T = torch.cat(test_T, dim=0)
    test_OPR = torch.cat(test_OPR, dim=0)
    test_WPR = torch.cat(test_WPR, dim=0)
    return test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR

def train(lambd, weight_decay):

    device = torch.device('cpu')
    model = PINNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5000000, min_lr=1e-6)
    model_name = f'PINN_Minus_40_train_10_test_lr_0.01_f_loss_{lambd}_weight_decay_{weight_decay}'
    writer = SummaryWriter(f"./logs/{model_name}/")

    best_test_loss = float('inf')

    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(40)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(40, 50)))
    # test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_test_data()
    for epoch in range(100000):
        model.train()
        train_f_losses = []
        train_mse_losses = []
        train_losses = []
        
        f_loss, mse_loss = model.loss_fn(
            train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
        train_loss = f_loss * lambd + mse_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)

        train_f_losses.append(f_loss.item())
        train_mse_losses.append(mse_loss.item())
        train_losses.append(train_loss.item())
        
        train_f_loss = np.mean(train_f_losses)
        train_mse_loss = np.mean(train_mse_losses)
        train_loss = np.mean(train_losses)

        if epoch % 5 == 0:
            model.eval()
            test_f_losses = []
            test_mse_losses = []
            test_losses = []

            test_f_loss, test_mse_loss = model.loss_fn(
                test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
            test_loss = test_f_loss * lambd + test_mse_loss
            
            test_f_losses.append(test_f_loss.item())
            test_mse_losses.append(test_mse_loss.item())
            test_losses.append(test_loss.item())
            
            test_f_loss = np.mean(test_f_losses)
            test_mse_loss = np.mean(test_mse_losses)
            test_loss = np.mean(test_losses)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if not os.path.exists('./TrainedModel'):
                    os.makedirs('./TrainedModel')
                torch.save(model.state_dict(), f'./TrainedModel/{model_name}.pth')
            # Logging the losses to TensorBoard
            
            print(epoch, f'train_loss:{train_loss.item():.5f}', f"train_f_loss:{train_f_loss.item():5f}", f"train_mse_loss:{train_mse_loss.item():5f}")

            # Log train losses
            writer.add_scalars('Loss/F Loss', {'Train': train_f_loss.item(), 'Test': test_f_loss.item()}, epoch)
            writer.add_scalars('Loss/MSE Loss', {'Train': train_mse_loss.item(), 'Test': test_mse_loss.item()}, epoch)
            writer.add_scalars('Loss/Total Loss', {'Train': train_loss.item(), 'Test': test_loss.item()}, epoch)
            writer.add_scalars('Learning Rate', {'Learning Rate': optimizer.param_groups[0]['lr']}, epoch)

            print(epoch, f"test_loss:{test_loss.item():.5f}", f"test_f_loss:{test_f_loss.item():5f}", f"test_mse_loss:{test_mse_loss.item():5f}")
    writer.close()


def trainPureML():

    model = PureMachineLearning().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5000000, min_lr=1e-6)
    model_name = f'PureML/PureMachineLearningModel_epoch_1000000'
    writer = SummaryWriter(f"./logs_pureml/{model_name}/")
    best_test_loss = float('inf')

    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(40)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(40, 50)))
    for epoch in range(1000000):
        model.train()
        train_mse_losses = []
        train_losses = []
        
        mse_loss = model.loss_fn(train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
        train_loss =  mse_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)

        train_mse_losses.append(mse_loss.item())
        train_losses.append(train_loss.item())
        
        train_mse_loss = np.mean(train_mse_losses)
        train_loss = np.mean(train_losses)

        if epoch % 5 == 0:
            model.eval()
            test_mse_losses = []
            test_losses = []

            test_mse_loss = model.loss_fn(
                test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
            test_loss = test_mse_loss
            
            test_mse_losses.append(test_mse_loss.item())
            test_losses.append(test_loss.item())
            
            test_mse_loss = np.mean(test_mse_losses)
            test_loss = np.mean(test_losses)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if not os.path.exists(os.path.dirname(f'./TrainedModel/{model_name}.pth')):
                    os.makedirs(os.path.dirname(f'./TrainedModel/{model_name}.pth'))   
                torch.save(model.state_dict(), f'./TrainedModel/{model_name}.pth')
            # Logging the losses to TensorBoard
            
            print(epoch, f'train_loss:{train_loss.item():.5f}', f"train_mse_loss:{train_mse_loss.item():5f}")

            # Log train losses
            writer.add_scalars('Loss/MSE Loss', {'Train': train_mse_loss.item(), 'Test': test_mse_loss.item()}, epoch)
            writer.add_scalars('Learning Rate', {'Learning Rate': optimizer.param_groups[0]['lr']}, epoch)

            print(epoch, f"test_loss:{test_loss.item():.5f}", f"test_mse_loss:{test_mse_loss.item():5f}")
    writer.close()


def train_test(lambd, weight_decay, lr, model_name):

    model = PINNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5000000, min_lr=1e-6)

    model_name = model_name
    writer = SummaryWriter(f"./new_logs_05/{model_name}/")

    best_test_loss = float('inf')

    train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR = load_all_training_data(list(range(40)))
    test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR = load_all_training_data(list(range(40, 50)))

    for epoch in range(100000):
        model.train()
        train_f_losses = []
        train_mse_losses = []
        train_losses = []
        
        f_loss, mse_loss = model.loss_fn(
            train_water_inject_rate, train_producer_bottom_pressure, train_T, train_OPR, train_WPR)
        train_loss = f_loss * lambd + mse_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step(train_loss)

        train_f_losses.append(f_loss.item())
        train_mse_losses.append(mse_loss.item())
        train_losses.append(train_loss.item())
        
        train_f_loss = np.mean(train_f_losses)
        train_mse_loss = np.mean(train_mse_losses)
        train_loss = np.mean(train_losses)

        if epoch % 5 == 0:
            model.eval()
            test_f_losses = []
            test_mse_losses = []
            test_losses = []

            test_f_loss, test_mse_loss = model.loss_fn(
                test_water_inject_rate, test_producer_bottom_pressure, test_T, test_OPR, test_WPR)
            test_loss = test_f_loss * lambd + test_mse_loss
            
            test_f_losses.append(test_f_loss.item())
            test_mse_losses.append(test_mse_loss.item())
            test_losses.append(test_loss.item())
            
            test_f_loss = np.mean(test_f_losses)
            test_mse_loss = np.mean(test_mse_losses)
            test_loss = np.mean(test_losses)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if not os.path.exists('./NewTrainedModel'):
                    os.makedirs('./NewTrainedModel')
                torch.save(model.state_dict(), f'./NewTrainedModel/{model_name}.pth')
            # Logging the losses to TensorBoard
            
            print(epoch, f'train_loss:{train_loss.item():.5f}', f"train_f_loss:{train_f_loss.item():5f}", f"train_mse_loss:{train_mse_loss.item():5f}")

            # Log train losses
            writer.add_scalars('Loss/F Loss', {'Train': train_f_loss.item(), 'Test': test_f_loss.item()}, epoch)
            writer.add_scalars('Loss/MSE Loss', {'Train': train_mse_loss.item(), 'Test': test_mse_loss.item()}, epoch)
            writer.add_scalars('Loss/Total Loss', {'Train': train_loss.item(), 'Test': test_loss.item()}, epoch)
            writer.add_scalars('Learning Rate', {'Learning Rate': optimizer.param_groups[0]['lr']}, epoch)

            print(epoch, f"test_loss:{test_loss.item():.5f}", f"test_f_loss:{test_f_loss.item():5f}", f"test_mse_loss:{test_mse_loss.item():5f}")
    writer.close()



if __name__ == '__main__':
    # trainPureML()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', default=0.1, type=float, required=False)
    parser.add_argument('--weight_decay', default=1e-5,type=float, required=False)
    parser.add_argument('--lr', default=0.005,type=float, required=False)
    parser.add_argument('--expid', default=0,type=str, required=False)

    args = parser.parse_args()
    lambd = args.lambd
    weight_decay = args.weight_decay
    lr = args.lr
    expid = args.expid

    model_name = f'expid_{expid}_lr_{lr}_f_loss_{lambd}_weight_decay_{weight_decay}'
    # train_test(lambd = lambd, weight_decay = weight_decay, lr = lr, model_name = model_name)
    train(lambd = lambd, weight_decay = weight_decay)

