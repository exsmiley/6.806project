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
from cnn import *

dev_data = AndroidDataSet(test=False)
test_data = AndroidDataSet(test=True)
model = torch.load('epoch1cnn.pt')

batch_size = 32
dropout_prob = 0.1
learning_rate = 0.001
margin = 0.2
momentum = 0
num_epoches = 1
num_hidden = 667
emb_size = 200


params = {'num_epoches' : num_epoches, 'momentum' : momentum,
            'margin' : margin, 'learning_rate' : learning_rate,
            'batch_size' : batch_size}

print('Started Evaluating on Dev Set...\n')

is_training = False

run_model(dev_data, None, model, is_training, params)

print('\nFinished Evaluating on Dev Set!\n')


print('Started Evaluating on Test Set...\n')

is_training = False

run_model(test_data, None, model, is_training, params)

print('\nFinished Evaluating on Test Set!\n')
