import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from prepare import *
import tqdm
from max_margin_loss import max_margin_loss
from evaluation import Evaluation
from lstm import *


class QA_CNN(nn.Module):

    def __init__(self, num_hidden, emb_size, embeddings, dropout_prob):
        super(QA_CNN, self).__init__()
        self.num_hidden = num_hidden
        self.emb_size = emb_size
        self.embeddings = embeddings
        self.embedding_layer = nn.Embedding(1, self.emb_size)
        self.embedding_layer.weight.data = torch.from_numpy(self.embeddings).float()
        self.dropout = nn.Dropout(dropout_prob)

        self.conv1 = nn.Conv1d(emb_size, num_hidden, 3, padding=1)
        # self.conv2 = nn.Conv1d(6, 16, 5)
        # an affine operation: y = Wx + b

    def forward(self, sent, mask):
        x = self.embedding_layer(sent.long())
        # batch_size, length = sent.size()
        x = torch.transpose(x, 2, 1)
        x = F.relu(self.dropout(self.conv1(x)))
        x = torch.transpose(x, 2, 1)
        masked = torch.mul(x, mask.unsqueeze(2).expand_as(x))
        sum_out = masked.sum(dim=1)
        count = mask.sum(dim=1).unsqueeze(1).expand_as(sum_out)
        x =  torch.div(sum_out, count)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    batch_size = 32
    dropout_prob = 0.1
    learning_rate = 0.01
    margin = 0.2
    momentum = 0.9
    num_epoches = 50
    num_hidden = 667
    emb_size = 200


    params = {'num_epoches' : num_epoches, 'momentum' : momentum,
                'margin' : margin, 'learning_rate' : learning_rate,
                'batch_size' : batch_size}

    train_data = UbuntuSequentialDataSet('ubuntu_data/train_random.txt')
    dev_data = UbuntuEvaluationDataSet('ubuntu_data/dev.txt')
    test_data = UbuntuEvaluationDataSet('ubuntu_data/test.txt')

    model = QA_CNN(num_hidden, emb_size, embeddings, dropout_prob)

    print('Started Training...\n')

    is_training = True

    run_model(train_data, dev_data, model, is_training, params)

    print('\nFinished Training!\n')

    print('Started Evaluating on Dev Set...\n')

    is_training = False

    run_model(test_data, model, is_training, params)

    print('\nFinished Evaluating on Dev Set!\n')

if __name__ == '__main__':
    main()
