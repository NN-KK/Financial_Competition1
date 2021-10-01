import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset

COL = ['open', 'high', 'low', 'close']
LAG = 5

class FinancialDataset(Dataset):
    def __init__(self, df, windows, task):
        if task == 'price':
            df['target'] = df['close'].shift(-1) / df['close']
        df.dropna(inplace=True)
        data = df.loc[:, COL]
        target = df.loc[:, 'target']
        data = self.minmax_norm(data).values
        target = target.values
        #target = self.minmax_norm(target).values
        data_list, target_list = [], []
        for i in range(data.shape[0] - windows -1):
            datas = []
            for j in range(data.shape[1]):
                datas.append(data[i:i+windows,j])
            data_list.append(datas)
            target_list.append(target[i+windows])
        self.x = torch.tensor(np.array(data_list).astype(np.float32), requires_grad=True).unsqueeze(1)
        self.y = torch.tensor(np.array(target_list).astype(np.float32))
      
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def minmax_norm(self, df):
        return (df - df.min()) / (df.max() - df.min())
    
    def create_logret(self, df, cols):
        for col in cols:
            logret_col = col + '_logret'
            df[logret_col] = np.log(df[col]/df[col].shift(1))

    def create_lags(self, df, cols):
        for col in cols:
            data = df.loc[:, col]
            logret_col = col + '_logret'
            for lag in range(1, LAG+1):
                lag_col = f'{logret_col}_lag{lag}'
                df[lag_col] = df[logret_col].shift(lag)
