import torch.nn as nn
from torch.nn.modules import dropout

class GRU(nn.Module):
    def __init__(self, output_dim = 1, hidden_dim = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(4, hidden_dim, batch_first = True, bidirectional = True, dropout = 0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, gru_out = self.gru(x)
        output = self.linear(gru_out[0].view(-1, self.hidden_dim))
        return self.sigmoid(output.squeeze())