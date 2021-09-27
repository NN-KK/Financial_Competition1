import math
import torch
import torch.nn as nn
from torch.nn.modules import dropout

SEED = 42

torch.manual_seed(SEED)

class LSTM(nn.Module):
    def __init__(self, output_dim = 1, hidden_dim_1 = 30, hidden_dim_2 = 50, dropout = 0.5):
        super().__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.lstm_1 = nn.LSTM(4, 
                            hidden_dim_1,
                            batch_first=True)
        self.drop_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(hidden_dim_1, 
                            hidden_dim_2,
                            batch_first=True)
        self.drop_2 = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim_2, output_dim)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.lstm_1.weight_ih_l0, std=1/math.sqrt(4))
        nn.init.normal_(self.lstm_1.weight_hh_l0, std=1/math.sqrt(hidden_dim_1))
        nn.init.zeros_(self.lstm_1.bias_ih_l0)
        nn.init.zeros_(self.lstm_1.bias_hh_l0)

        nn.init.normal_(self.lstm_2.weight_ih_l0, std=1/math.sqrt(hidden_dim_1))
        nn.init.normal_(self.lstm_2.weight_hh_l0, std=1/math.sqrt(hidden_dim_2))
        nn.init.zeros_(self.lstm_2.bias_ih_l0)
        nn.init.zeros_(self.lstm_2.bias_hh_l0)

    def forward(self, x):
        lstm_1_out, hidden_1 = self.lstm_1(x)
        lstm_1_out = self.drop_1(lstm_1_out)
        lstm_2_out, hidden_2 = self.lstm_2(lstm_1_out)
        output = self.linear(hidden_2[0].view(-1, self.hidden_dim_2))
        return self.sigmoid(output.squeeze())

