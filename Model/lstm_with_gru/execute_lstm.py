import warnings
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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

sys.path.append('.')

from Model.lstm_with_gru.dataset.dataset import FinancialDataset
from Model.lstm_with_gru.model.lstm import LSTM
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
        return torch.utils.data.DataLoader(FinancialDataset(df[:train_size], windows), batch_size = batch_size)
    elif type == 'val':
        return torch.utils.data.DataLoader(FinancialDataset(df[train_size:], windows), batch_size = batch_size)

def main(file_name, train_ratio, batch_size, windows, n_epochs):
    train_loader = dataloader(file_name, train_ratio, batch_size, windows, 'train')
    val_loader = dataloader(file_name, train_ratio, batch_size, windows, 'val')

    model = LSTM()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay=1e-2)
    loss_fn = nn.BCELoss()
    
    all_loss_train = 0
    all_loss_val = 0

    for epoch in range(n_epochs):
        for batch, target in train_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss = loss_fn(output.view(-1), target)
            #loss = -roc_auc_score(target.detach().numpy().copy(), output.view(-1).detach().numpy().copy())
            #print(loss)
            #loss = torch.from_numpy(loss.astype(np.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss_train += loss.item()
        for batch, target in val_loader:
            batch_size = batch.shape[0]
            output = model(batch.view(batch_size, windows, 4))
            loss = loss_fn(output.view(-1), target)
            #loss = -roc_auc_score(target.detach().numpy().copy(), output.view(-1).detach().numpy().copy())
            #loss = torch.from_numpy(loss.astype(np.float32))
            all_loss_val += loss.item()
        if epoch % 10 == 9:
            print("epoch", epoch+1, "\t" , "loss", float(all_loss_train/(len(train_loader)*10)), float(all_loss_val/(len(val_loader)*10)))
            all_loss_train = 0
            all_loss_val = 0
    return predict(file_name, windows, model)

DIR_PATH = "./Pool/"
file_names = os.listdir(DIR_PATH)
# 銘柄の名前を入れるリスト
list_name = []
# 予測対象日結果を入れるリスト
list_predict = []
list_target = []
i = 0
for file_name in file_names:
    print(str(i), " : ", file_name)
    i += 1
    executed = main(file_name, 0.7, 16, 7, 50)
    predicted = executed[0].view(-1).item()
    target = float(executed[1])
    print(predicted, target)
    # 銘柄の名前の追加
    list_name.append(file_name.split('_')[2][:-4])
    # 予測対象日結果の追加
    list_predict.append(predicted)
    list_target.append(target)

df = pd.DataFrame(list_predict, columns = ["result"] , index = list_name)
df.to_csv('predict_1_lstm.csv')
df2 = pd.DataFrame(list_target, columns = ["target"] , index = list_name)
df_merge = pd.concat([df, df2['target']], axis = 1)
df_merge.to_csv('predict_target_1_lstm.csv')

print(roc_auc_score(list_target, list_predict))
