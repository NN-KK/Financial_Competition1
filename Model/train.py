import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim

from Data.make_data import execute
from Model.dataset.dataset import FinancialDataset
from Model.rfc.model import RFC

SEED = 42

def dataloader(file_name, train_ratio, batch_size, type):
    df = pd.read_csv('./Pool/ohlc_' + file_name + '.csv')
    n_samples = df.shape[0]
    train_size = int(n_samples * train_ratio)
    if type == 'train':
        return torch.utils.data.DataLoader(FinancialDataset(df[:train_size]), batch_size = batch_size)
    else:
        return torch.utils.data.DataLoader(FinancialDataset(df[train_size:]), batch_size = batch_size)

def training_loop(file_name, train_ratio, batch_size, learning_rate, n_epochs):
    torch.manual_seed(SEED)
    train_loader = dataloader(file_name, train_ratio, batch_size, batch_size, 'train')
    val_loader = dataloader(file_name, train_ratio, batch_size, batch_size, 'val')

    model = RFC()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    loss_fn = nn.BCELoss()

    #for epoch in range(n_epochs):


