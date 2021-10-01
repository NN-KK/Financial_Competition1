import warnings
warnings.simplefilter('ignore')

import sys
import os
import pandas as pd
import numpy as np
import ast
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
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

def main(model, optimizer, file_name, train_ratio, batch_size, windows, n_epochs):
    train_loader = dataloader(file_name, train_ratio, batch_size, windows, 'train')
    val_loader = dataloader(file_name, train_ratio, batch_size, windows, 'val')

    #optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay=1e-4)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        all_loss_train = 0
        all_loss_val = 0
        all_roc_auc = 0
        for batch, target in train_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss_train = loss_fn(output.view(-1), target)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            all_loss_train += loss_train.item()
        for batch, target in val_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss_valid = loss_fn(output.view(-1), target)
            all_loss_val += loss_valid.item()
            all_roc_auc += mape(target.detach().numpy().copy(), output.view(-1).detach().numpy().copy())
        if epoch % 10 == 9:
            print("epoch", epoch+1, "\t" , "loss", float(all_loss_train/(len(train_loader))), float(all_loss_val/(len(val_loader))), float(all_roc_auc/len(val_loader)))
            

    return predict(file_name, windows, model, 'price')
    #return output[-2], target[-2]

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)
# 銘柄の名前を入れるリスト
list_name = []
# 予測対象日結果を入れるリスト
list_predict = []
list_target = []
i = 1
param_csv = pd.read_csv('parameter_price_0929.csv')
val = 0
for file_name in file_names:
    param = param_csv[param_csv['file_name'] == file_name]['best_parameter'].values[0]
    param = ast.literal_eval(param)

    hl1 = int(param['hidden_dim_lstm_1'])
    hl2 = int(param['hidden_dim_lstm_2'])
    hg = int(param['hidden_dim_gru'])
    d = param['dropout']
    model = LSTM_GRU(task='price', hidden_dim_lstm_1 = hl1, hidden_dim_lstm_2 = hl2, hidden_dim_gru = hg, dropout = d)
    
    if param['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = param['adam_lr'], weight_decay=param['weight_decay'])
    elif param['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=param['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr = param['momentum_sgd_lr'], weight_decay=param['weight_decay'])

    batch_size = int(param['batch_size'])
    windows = int(param['windows'])

    print(str(i), " : ", file_name)
    i += 1
    executed = main(model, optimizer, file_name, 0.7, batch_size, windows, 100)
    predicted = executed[0].view(-1).item()
    target = float(executed[1])
    val += float(executed[2])
    print(predicted, target)
    # 銘柄の名前の追加
    list_name.append(file_name.split('_')[2][:-4])
    # 予測対象日結果の追加
    list_predict.append(predicted)
    list_target.append(target)

df = pd.DataFrame(list_predict, columns = ["result"] , index = list_name)
df.to_csv('predict_2_lstm_with_gru.csv')
df2 = pd.DataFrame(list_target, columns = ["target"] , index = list_name)
df_merge = pd.concat([df, df2['target']], axis = 1)
df_merge.to_csv('predict_target_2_lstm_with_gru.csv')

print(mape(np.array(list_target), np.array(list_predict)), val/22)
