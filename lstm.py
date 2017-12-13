import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from prepare import *
import tqdm

class QA_LSTM(nn.Module):

    def __init__(self, num_hidden, emb_size, embeddings, dropout_prob):
        super(QA_LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.emb_size = emb_size
        self.embeddings = embeddings

        self.lstm = nn.LSTM(emb_size, num_hidden)
        self.dropout = nn.Dropout(dropout_prob)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.num_hidden)),
                autograd.Variable(torch.zeros(1, 1, self.num_hidden)))

    def forward(self, x):
        self.embedding_layer = nn.Embedding(x.shape()[0], self.emb_size)
        self.embedding_layer.weight.data = torch.from_numpy(self.embeddings)
        x = self.embedding_layer(x)
        lstm_out, self.hidden = self.lstm(x)
        out = self.dropout(lstm_out)
        return out



def model_size():
    net = QA_LSTM()
    params = list(net.parameters())
    total = 0
    for param in params:
        this = 1
        for p in param.size():
            this *= p
        total += this
    return total


def run_model(data, model, is_training, params):
     # Does the update
    learning_rate = params['learning_rate']
    margin = params['margin']
    momentum = params['momentum']
    num_epoches = params['num_epoches']

    loss_func = nn.MarginRankingLoss(margin)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in tqdm.tqdm(range(num_epoches)):  # loop over the dataset multiple times
        print "Running epoch:", epoch+1
        run_epoch(data, model, optimizer, is_training, params)
        if i % 5 == 0:
            torch.save(net, 'mytraining{}.pt'.format(i))
    torch.save(net, 'mytraininglast.pt')

def run_epoch(data, model, optimizer, is_training, params):

    batch_size = params['batch_size']
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    running_loss = 0.0
    for i, data in enumerate(loader):
        # get the inputs
        question, candidate_set = data

        # wrap them in Variable
        question_title, question_body = Variable(question[0]), Variable(question[1])

        question_title, question_body = Variable(question[0]), Variable(question[1])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1 = net(input1).float()
        output2 = net(input2).float()
        # print output1, output2, labels
        # labels = Variable(torch.ones(output1.size()[1]))
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]


def main():

    batch_size = 32
    dropout_prob = 0.1
    learning_rate = 1e-3
    margin = 0.2
    momentum = 0.9
    num_epoches = 50
    num_hidden = 400
    emb_size = 200
    is_training = True

    params = {'num_epoches' : num_epoches, 'momentum' : momentum,
                'margin' : margin, 'learning_rate' : learning_rate,
                'batch_size' : batch_size}

    train_data = UbuntuSequentialDataSet('ubuntu_data/train_random.txt')
    dev_data = UbuntuSequentialDataSet('ubuntu_data/dev.txt')
    test_data = UbuntuSequentialDataSet('ubuntu_data/test.txt')




    model = QA_LSTM(num_hidden, emb_size, embeddings, dropout_prob)

    print('Started Training...\n')

    run_model(train_data, model, is_training, params)

    print('\nFinished Training!\n')

    print('Started Evaluating...\n')

    run_model(test_data, model, is_training, params)

    print('\nFinished Evaluating!\n')

if __name__ == '__main__':
    main()
    # model_size()
