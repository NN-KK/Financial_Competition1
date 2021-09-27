import pandas as pd
import numpy as np
import torch

COL = ['open', 'high', 'low', 'close']

def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())

def predict(file_name, windows, model, task):
    df = pd.read_csv('./Pool/' + file_name)
    data = df.loc[:, COL]
    if task == 'updown':
        target = df.loc[:, 'target']
    else:
        target = df['close'].shift(-1) / df['close']
        target = target.values
    data = minmax_norm(data).values
    #target = minmax_norm(target).values
    data_list, target_list = [], []
    for i in range(data.shape[0] - windows -1):
        datas = []
        for j in range(data.shape[1]):
            datas.append(data[i:i+windows,j])
        data_list.append(datas)
        target_list.append(target[i+windows])
    x_test = torch.tensor(np.array(data_list[-2]).astype(np.float32), requires_grad=True).unsqueeze(1)
    t_test = torch.tensor(np.array(target_list[-2]).astype(np.float32))
    predict = model(x_test.view(1, windows, 4))
    return predict, t_test