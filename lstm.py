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

    for epoch in tqdm.tqdm(range(num_epoches)):
        print "Running epoch:", epoch + 1
        run_epoch(data, model, optimizer, loss_func, is_training, params)
        if i % 5 == 0:
            torch.save(net, 'mytraining{}.pt'.format(i))
    torch.save(net, 'mytraininglast.pt')

def run_epoch(data, model, optimizer, loss_func, is_training, params):

    batch_size = params['batch_size']
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    running_loss = 0.0

    model.train() if is_training else model.eval()


    for i, data in enumerate(loader):
        # get the inputs

        if is_training:
            q_title, q_body, c_titles, c_bodies, labels = data
        else:
            question, candidate_set, vector, bm25s = data


        # wrap them in Variable
        q_title, q_body = Variable(q_title), Variable(q_body)

        c_titles, c_bodies = Variable(c_titles), Variable(c_bodies)

        # zero the parameter gradients
        if is_training:
            optimizer.zero_grad()

        q_enc = (model(q_body) + model(q_title)) / 2
        c_enc = (model(c_titles) + model(c_bodies)) / 2

        cos_sim = torch.nn.CosineSimilarity(2)
        sims = cos_sim(q_enc, c_enc)



        if is_training:
            loss = loss_func(sims, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        else:
            pass



def main():

    batch_size = 32
    dropout_prob = 0.1
    learning_rate = 1e-3
    margin = 0.2
    momentum = 0.9
    num_epoches = 50
    num_hidden = 400
    emb_size = 200


    params = {'num_epoches' : num_epoches, 'momentum' : momentum,
                'margin' : margin, 'learning_rate' : learning_rate,
                'batch_size' : batch_size}

    train_data = UbuntuSequentialDataSet('ubuntu_data/train_random.txt')
    dev_data = UbuntuEvaluationDataSet('ubuntu_data/dev.txt')
    test_data = UbuntuEvaluationDataSet('ubuntu_data/test.txt')




    model = QA_LSTM(num_hidden, emb_size, embeddings, dropout_prob)

    print('Started Training...\n')

    is_training = True

    run_model(train_data, model, is_training, params)

    print('\nFinished Training!\n')

    print('Started Evaluating on Dev Set...\n')

    is_training = False

    run_model(test_data, model, is_training, params)

    print('\nFinished Evaluating on Dev Set!\n')

if __name__ == '__main__':
    main()
    # model_size()
