import warnings

from torch.optim import optimizer
from torch.utils import data
warnings.simplefilter('ignore')

import sys
import os
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
import optuna
from optuna import Trial
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

sys.path.append('.')

from Model.lstm_with_gru.dataset.dataset import FinancialDataset
from Model.lstm_with_gru.model.lstm_with_gru import LSTM_GRU
from Model.lstm_with_gru.predict import predict

SEED = 42
LAG = 5
COL = ['open', 'high', 'low', 'close']

torch.manual_seed(SEED)

def mape(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)
    return output_errors

def dataloader(file_name, train_ratio, batch_size, windows, type):
    df = pd.read_csv('./Pool/' + file_name)
    n_samples = df.shape[0]
    train_size = int(n_samples * train_ratio)
    if type == 'train':
        return torch.utils.data.DataLoader(FinancialDataset(df[:train_size], windows, 'price'), batch_size = batch_size)
    elif type == 'val':
        return torch.utils.data.DataLoader(FinancialDataset(df[train_size:], windows, 'price'), batch_size = batch_size)

def train(model, file_name, optimizer, batch_size, windows, train_ratio = 0.7):
    model.train()
    loss_fn = oss_fn = nn.MSELoss()
    train_loader = dataloader(file_name, train_ratio, batch_size, windows, 'train')
    for batch, target in train_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss_train = loss_fn(output.view(-1), target)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

def valid(model, file_name, batch_size, windows, train_ratio = 0.7):
    model.eval()
    loss_fn = nn.MSELoss()
    all_loss_val = 0
    all_roc_auc = 0
    val_loader = dataloader(file_name, train_ratio, batch_size, windows, 'val')
    with torch.no_grad():
        for batch, target in val_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss_valid = loss_fn(output.view(-1), target)
            all_loss_val += loss_valid.item()
    return all_loss_val/len(val_loader)

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    
    if optimizer_name == optimizer_names[0]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters())
    
    return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)
    
    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu
    
    return activation

def objective(trial):
    #file_name = 'ohlc_Binance_ADAUSDT_d.csv'
    hl1 = int(trial.suggest_discrete_uniform('hidden_dim_lstm_1', 10, 100, 10))
    hl2 = int(trial.suggest_discrete_uniform('hidden_dim_lstm_2', 10, 100, 10))
    hg = int(trial.suggest_discrete_uniform('hidden_dim_gru', 10, 100, 10))
    d = trial.suggest_loguniform('dropout', 0.1, 0.5)
    model = LSTM_GRU(task='price', hidden_dim_lstm_1=hl1, hidden_dim_lstm_2=hl2, hidden_dim_gru=hg, dropout=d)
    optimizer = get_optimizer(trial, model)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))
    windows = int(trial.suggest_discrete_uniform('windows', 1, 7, 2))
    for epoch in range(10):
        train(model, file_name, optimizer, batch_size, windows)
        mse = valid(model, file_name, batch_size, windows)
    return mse

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)

best_parameter = pd.DataFrame(columns = ['file_name', 'best_parameter'])
i = 1
for file_name in file_names:
    TRIAL_SIZE = 100
    print('--------------------------------------------------------------------------------')
    print(i, file_name)
    i += 1
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=TRIAL_SIZE)

    best_parameter = best_parameter.append({'file_name': file_name, 'best_parameter': study.best_params}, ignore_index=True)

best_parameter.to_csv('parameter.csv', index=False)
