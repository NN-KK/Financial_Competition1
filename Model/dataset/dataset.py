import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Data.make_data import execute

COL = ['open', 'high', 'low', 'close']

class FinancialDataset(Dataset):
    def __init__(self, file_name):
        df = execute(file_name)
        self.x = df.loc[:, COL]
        self.y = df.loc[:, 'target']
      
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]