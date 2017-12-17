import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from prepare import *
import tqdm


class DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DAN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax()

    def forward(self, inp):
        x = F.relu(self.hidden(inp))
        x = self.output(x)
        return self.softmax(x)


def train_dan(encoder_model):

    input_size = 667
    hidden_size = 200
    num_epochs = 10
    learning_rate = 0.01
    weight_decay = 0.1
    batch_size = 128

    # make the model
    adv_model = torch.load('adv_model.pt')#DAN(input_size, hidden_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(adv_model.parameters(), lr=learning_rate, weight_decay=weight_decay)  

    print('Starting training...')

    for j in xrange(num_epochs):
        print('Running epoch {}...'.format(j))
        dataset = DomainDataSet()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for i, data in tqdm.tqdm(enumerate(loader)):
            if i != len(loader) - 1:
                t, t_mask, b, b_mask, label = data
                t, t_mask, b, b_mask, label = Variable(t), Variable(t_mask), Variable(b), Variable(b_mask), Variable(label).long()
                encoded = (encoder_model(t, t_mask) + encoder_model(b, b_mask))/2

                output = adv_model(encoded)

                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

    print('Saving model...')
    torch.save(adv_model, 'adv_model.pt')


def eval_model(encoder_model):
    adv_model = torch.load('adv_model.pt')
    print('Evaluating...')
    dataset = DomainDataSet()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    total = 0
    correct = 0.
    for i, data in tqdm.tqdm(enumerate(loader)):
        t, t_mask, b, b_mask, label = data
        t, t_mask, b, b_mask, label = Variable(t), Variable(t_mask), Variable(b), Variable(b_mask), label
        encoded = (encoder_model(t, t_mask) + encoder_model(b, b_mask))/2

        output = adv_model(encoded).data.numpy()[0]
        total += 1
        if np.argmax(output) == label.numpy():
            correct += 1
    print('Accuracy: {}'.format(correct/total))


def domain_training_encoder(encoder_model):
    '''trains the encoder model to work in either domain'''
    learning_rate = 0.0001
    lam = 1e-7
    weight_decay = 0.1
    adv_model = torch.load('adv_model.pt')#DAN(input_size, hidden_size)
    num_epochs = 10
    batch_size = 128

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)  

    print('Starting training...')

    for j in xrange(num_epochs):
        print('Running epoch {}...'.format(j))
        dataset = DomainDataSet()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for i, data in tqdm.tqdm(enumerate(loader)):
            if i != len(loader) - 1:
                t, t_mask, b, b_mask, label = data
                t, t_mask, b, b_mask, label = Variable(t), Variable(t_mask), Variable(b), Variable(b_mask), Variable(label).long()
                encoded = (encoder_model(t, t_mask) + encoder_model(b, b_mask))/2

                output = adv_model(encoded)

                optimizer.zero_grad()
                loss = -lam*criterion(output, label)
                loss.backward()
                optimizer.step()


    eval_model(encoder_model)
    # print('Saving model...')
    torch.save(encoder_model, 'enc2.pt')

    

if __name__ == '__main__':
    from cnn import *
    encoder_model = torch.load('epoch1cnn.pt')
    domain_training_encoder(encoder_model)
    