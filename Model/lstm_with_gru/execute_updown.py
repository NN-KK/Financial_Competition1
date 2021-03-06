import warnings
warnings.simplefilter('ignore')

import sys
import os
import pandas as pd
import numpy as np
import ast
from typing import Any, Callable, Optional
from sklearn.ensemble import RandomForestClassifier
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

def dataloader(file_name, train_ratio, batch_size, windows, type):
    df = pd.read_csv('./Pool/' + file_name)
    n_samples = df.shape[0]
    train_size = int(n_samples * train_ratio)
    if type == 'train':
        return torch.utils.data.DataLoader(FinancialDataset(df[:train_size], windows, 'updown'), batch_size = batch_size)
    elif type == 'val':
        return torch.utils.data.DataLoader(FinancialDataset(df[train_size:], windows, 'updown'), batch_size = batch_size)

def main(model, optimizer, file_name, train_ratio, batch_size, windows, n_epochs):
    train_loader = dataloader(file_name, train_ratio, batch_size, windows, 'train')
    val_loader = dataloader(file_name, train_ratio, batch_size, windows, 'val')

    #model = LSTM_GRU(task='updown')
    #optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay=1e-4)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    #rfc = RandomForestClassifier(random_state=SEED)
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
            try:
                all_roc_auc += roc_auc_score(target.detach().numpy().copy(), output.detach().numpy().copy())
            except ValueError as e:
                all_roc_auc += 0
        if epoch % 10 == 9:
            print("epoch", epoch+1, "\t" , "loss", float(all_loss_train/(len(train_loader))), float(all_loss_val/(len(val_loader))), float(all_roc_auc/len(val_loader)))
            #print("epoch", epoch+1, "\t" , "loss", float(loss_train/(len(train_loader))), float(loss_valid/(len(val_loader))), float(all_roc_auc/len(val_loader)))
            

    return predict(file_name, windows, model, 'updown')

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)
# ????????????????????????????????????
list_name = []
# ??????????????????????????????????????????
list_predict = []
list_target = []
i = 1
param_csv = pd.read_csv('parameter_updown_0929.csv')
for file_name in file_names:
    param = param_csv[param_csv['file_name'] == file_name]['best_parameter'].values[0]
    param = ast.literal_eval(param)

    hl1 = int(param['hidden_dim_lstm_1'])
    hl2 = int(param['hidden_dim_lstm_2'])
    hg = int(param['hidden_dim_gru'])
    d = param['dropout']
    model = LSTM_GRU(task='updown', hidden_dim_lstm_1 = hl1, hidden_dim_lstm_2 = hl2, hidden_dim_gru = hg, dropout = d)
    
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
    predicted = float(executed[0])
    target = float(executed[1])
    print(predicted, target)
    # ????????????????????????
    list_name.append(file_name.split('_')[2][:-4])
    # ??????????????????????????????
    list_predict.append(predicted)
    list_target.append(target)

df = pd.DataFrame(list_predict, columns = ["result"] , index = list_name)
df.to_csv('predict_1_lstm_with_gru.csv')
df2 = pd.DataFrame(list_target, columns = ["target"] , index = list_name)
df_merge = pd.concat([df, df2['target']], axis = 1)
df_merge.to_csv('predict_target_1_lstm_with_gru.csv')

print(roc_auc_score(list_target, list_predict))
