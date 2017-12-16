import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d
from prepare import *


class AdvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(AdvNet, self).__init__()
        self.in_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.in_to_hidden(x)
        x = F.relu(x)
        out = self.softmax(self.hidden_to_output(x))
        return out


def train_adv():
    # android_data = load_tokenized_text('android_data/corpus.tsv.gz', 'Android', return_index=False)


    input_dim = 10
    hidden_dim = 200

    model = AdvNet(input_dim, hidden_dim)

    # TODO do training

    torch.save(model, 'adv.pt')

