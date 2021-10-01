import sys
import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

sys.path.append('.')

from Model.dataset.dataset import FinancialDataset
from Model.sdae.sdae import StackedDenoisingAutoEncoder
import  Model.sdae.model as ae
SEED = 42

torch.manual_seed(SEED)

def dataloader(file_name, train_ratio, batch_size, type):
    df = pd.read_csv('./Pool/' + file_name)
    n_samples = df.shape[0]
    train_size = int(n_samples * train_ratio)
    if type == 'train':
        return torch.utils.data.DataLoader(FinancialDataset(df[:train_size]), batch_size = batch_size)
    else:
        return torch.utils.data.DataLoader(FinancialDataset(df[train_size:]), batch_size = batch_size)


def main(
        file_name,
        cuda = False, 
        batch_size = 10, 
        pretrain_epochs = 10, 
        finetune_epochs = 100, 
        testing_mode = False,  
        ):
    autoencoder = StackedDenoisingAutoEncoder(
        [4, 500, 500, 2000, 10], final_activation=None
    )
    ae.pretrain(
        file_name,
        autoencoder,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-5),
        #optimizer=lambda model: Adam(model.parameters(), lr=1e-4, weight_decay=1e-4),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
        cuda=cuda
    )

    

if __name__ == "__main__":
    main('ohlc_Binance_ADAUSDT_d.csv')