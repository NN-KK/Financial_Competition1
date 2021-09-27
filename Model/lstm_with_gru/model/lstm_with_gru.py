import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

SEED = 42

torch.manual_seed(SEED)

class LSTM_GRU(nn.Module):
    def __init__(self, task, hidden_dim_lstm_1 = 30, hidden_dim_lstm_2 = 30, hidden_dim_gru = 30, dropout = 0.5, output_dim = 1):
        super().__init__()
        self.task = task
        self.hidden_dim_lstm_2 = hidden_dim_lstm_1
        self.hidden_dim_lstm_2 = hidden_dim_lstm_2
        self.hidden_dim_gru = hidden_dim_gru
        self.lstm_1 = nn.LSTM(4, hidden_dim_lstm_1, batch_first=True)
        self.drop_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(hidden_dim_lstm_1, hidden_dim_lstm_2, batch_first=True)
        self.gru = nn.GRU(4, hidden_dim_gru, batch_first = True, bidirectional = True, dropout = 0.2)
        self.linear = nn.Linear(hidden_dim_lstm_2 + hidden_dim_gru, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_1_out, hidden_1 = self.lstm_1(x)
        lstm_1_out = self.drop_1(lstm_1_out)
        lstm_2_out, hidden_2 = self.lstm_2(lstm_1_out)
        gru_out, hidden = self.gru(x)
        hidden = self.drop_1(hidden[0])
        lstm_feature = hidden_2[0].view(-1, self.hidden_dim_lstm_2)
        gru_feature = hidden.view(-1, self.hidden_dim_gru)
        hybrid = torch.cat((lstm_feature, gru_feature), 1)
        output = self.linear(hybrid.view(-1, self.hidden_dim_lstm_2 + self.hidden_dim_gru))
        if self.task == 'updown':
            return self.sigmoid(output)
        else:
            return self.relu(output)