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

        self.conv1 = nn.Conv1d(emb_size, num_hidden, 3, stride=3)
        # self.conv2 = nn.Conv1d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(12800, num_hidden)
        self.fc2 = nn.Linear(num_hidden, emb_size)

    def forward(self, sent, mask):
        x = self.embedding_layer(sent.long())
        # x = self.dropout(sent)
        # if x.size()[1] == 40:
        #     print('hi mom')
        #     x = F.pad(x, (2, 60), "reflect", 0)
        x = x.view(32, 200, -1)
        # print('hi')
        print(x.size())
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.avg_pool1d(x, 2)
        # If the size is a square you can only specify a single number
        # x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    num_hidden = 800
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
    