import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tqdm

class DAN(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(DAN, self).__init__()
        self.hidden = nn.Linear(emb_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax()

    def forward(self, inp):
        x = F.relu(self.hidden(inp))
        x = self.output(x)
        return self.softmax(x)
