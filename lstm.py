import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from prepare import *
import tqdm
from max_margin_loss import max_margin_loss
from evaluation import Evaluation

class QA_LSTM(nn.Module):

    def __init__(self, num_hidden, emb_size, embeddings, dropout_prob):
        super(QA_LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.emb_size = emb_size
        self.embeddings = embeddings
        self.embedding_layer = nn.Embedding(1, self.emb_size)
        self.embedding_layer.weight.data = torch.from_numpy(self.embeddings).float()

        self.lstm = nn.LSTM(emb_size, num_hidden)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, sent, mask):
        sent = self.embedding_layer(sent.long())
        x = self.dropout(sent)
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        masked = torch.mul(lstm_out, mask.unsqueeze(2).expand_as(lstm_out))
        sum_out = masked.sum(dim=1)
        count = mask.sum(dim=1).unsqueeze(1).expand_as(sum_out)
        return torch.div(sum_out, count)



def model_size(model):
    params = list(model.parameters())
    total = 0
    for param in params:
        this = 1
        for p in param.size():
            this *= p
        total += this
    return total


def run_model(train_data, dev_data, model, is_training, params):
     # Does the update
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    num_epoches = params['num_epoches']


    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epoches):
        print("-------\nEpoch {}:\n".format(epoch))
        if is_training:
            loss = run_epoch(train_data, model, optimizer, is_training, params)

            if epoch % 5 == 0:
                map, mrr, p_at_one, p_at_five = run_epoch(dev_data, model, optimizer, False, params)
                print("MAP: {0}, MRR: {1}, P@1: {2}, P@5: {3}".format(map, mrr, p_at_one, p_at_five))
        else:
            map, mrr, p_at_one, p_at_five = run_epoch(train_data, model, optimizer, is_training, params)
            print("MAP: {0}, MRR: {1}, P@1: {2}, P@5: {3}".format(map, mrr, p_at_one, p_at_five))
        print('Train Loss: {}'.format(loss))



def run_epoch(data, model, optimizer, is_training, params):

    batch_size = params['batch_size']
    margin = params['margin']
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    running_loss = 0.0

    model.train() if is_training else model.eval()

    for i, data in tqdm.tqdm(enumerate(loader)):
        # get the inputs

        q_title, q_title_mask, q_body, q_body_mask, c_titles, c_titles_mask, c_bodies, c_bodies_mask, labels = data


        predicted = []
        # wrap them in Variable
        q_title, q_title_mask, q_body, q_body_mask = Variable(q_title), Variable(q_title_mask), Variable(q_body), Variable(q_body_mask)

        c_titles, c_titles_mask, c_bodies, c_bodies_mask = Variable(c_titles), Variable(c_titles_mask), Variable(c_bodies), Variable(c_bodies_mask)

        # zero the parameter gradients
        if is_training:
            optimizer.zero_grad()

        num_c = c_titles.size()[1]

        q_enc = (model(q_body, q_body_mask) + model(q_title, q_title_mask)) / 2

        c_enc = (model(c_titles.view(batch_size * num_c, 40), c_titles_mask.view(batch_size * num_c, 40)) +
                model(c_bodies.view(batch_size * num_c, 100), c_bodies_mask.view(batch_size * num_c, 100))) / 2

        q_enc = q_enc.view(batch_size, 1, -1).repeat(1, num_c, 1)
        c_enc = c_enc.view(batch_size, num_c, -1)

        cos_sim = torch.nn.CosineSimilarity(2)
        sims = cos_sim(q_enc, c_enc)

        if is_training:

            loss = max_margin_loss(sims, margin)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        else:
            s, i = sims.sort(dim=1, descending=True)
            predicted.extend([labels.data[x][i.data[x]] for x in range(batch_size)])

    return running_loss if is_training else Evaluation(predicted).evaluate()

def main():

    batch_size = 32
    dropout_prob = 0.1
    learning_rate = 0.1
    margin = 0.2
    momentum = 0.9
    num_epoches = 50
    num_hidden = 200
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

    run_model(train_data, dev_data, model, is_training, params)

    print('\nFinished Training!\n')

    print('Started Evaluating on Dev Set...\n')

    is_training = False

    run_model(test_data, model, is_training, params)

    print('\nFinished Evaluating on Dev Set!\n')

if __name__ == '__main__':
    main()
    # model_size()
