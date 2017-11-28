import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from prepare import *


class QA_CNN(nn.Module):

    def __init__(self):
        super(QA_CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(752, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def model_size():
    net = QA_CNN()
    print net
    params = list(net.parameters())
    total = 0
    for param in params:
        this = 1
        for p in param.size():
            this *= p
        total += this
    print total
    torch.save(net, 'test.pt')
    net2 = torch.load('test.pt')
    # print list(net2.parameters())[0]


def main():
    dataset = UbuntuDataSet()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    print "Loaded data..."
    net = QA_CNN()
    
    criterion = nn.MarginRankingLoss(1)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Does the update

    for epoch in range(100):  # loop over the dataset multiple times
        print "Running epoch:", epoch+1
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs
            input1, input2, labels = data

            # wrap them in Variable
            input1, input2, labels = Variable(input1), Variable(input2), Variable(labels)
            # input1.float()
            # input2.float()
            labels = labels.float()
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

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if epoch % 5 == 0:
            torch.save(net, 'mytraining{}.pt'.format(epoch))
    torch.save(net, 'mytraininglast.pt')

    print('Finished Training')

if __name__ == '__main__':
    main()
    # model_size()