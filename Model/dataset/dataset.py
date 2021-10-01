import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset

COL = ['open', 'high', 'low', 'close']

class FinancialDataset(Dataset):
    def __init__(self, df):
        data = df.loc[:, COL]
        target = df.loc[:, 'target']
        self.x = torch.tensor(data.values.astype(np.float32), requires_grad=True).unsqueeze(1)
        self.y = torch.tensor(target.values.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]