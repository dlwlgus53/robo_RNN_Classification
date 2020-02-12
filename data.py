import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

def getSeq_len(row):
    '''
    returns: count of non-nans (integer)
    adopted from: M4rtni's answer in stackexchange
    '''
    return np.count_nonzero(~np.isnan(row))

def getMask(batch):
    '''
    returns: boolean array indicating whether nans
    '''
    return (~np.isnan(batch)).astype(np.int32)

def trim_batch(batch):
    '''
    args: npndarray of a batch (bsz, n_features)
    returns: trimmed npndarray of a batch.
    '''
    max_seq_len = 0
    for n in range(batch.shape[0]):
        max_seq_len = max(max_seq_len, getSeq_len(batch[n]))

    if max_seq_len == 0:
        print("error in trim_batch()")
        sys.exit(-1)

    batch = batch[:,:max_seq_len]
    return batch

def batchify(data, bsz, labels):
    batches = []
    
    n_samples = data.shape[0]
    for n in range(0,n_samples,bsz):
        if n+bsz > n_samples: #discard remainder #TODO: use remainders somehow
            break
        batch = data[n:n+bsz]
        target = labels[n:n+bsz]

        batch = trim_batch(batch)
        mask = getMask(batch)

        batches.append([batch, mask, target])

    return batches

def get_batch(source, i):
    return torch.tensor(source[i][0]).type(torch.float32),\
           torch.tensor(source[i][1]).type(torch.float32),\
           torch.tensor(source[i][2]).type(torch.LongTensor)

def prepareData():
    df_train = pd.read_csv("robo_train2.csv")
    df_dummy = pd.read_csv("robo_dummy.csv")

    np_train = np.asarray(df_train)
    np_dummy = np.asarray(df_dummy)

    ones = np.ones((np_train.shape[0],1))
    zeros = np.zeros((np_dummy.shape[0],1))

    # concatconcat
    np_data = np.vstack([np_train, np_dummy])
    np_labels = np.vstack([ones, zeros])

    # shuffle
    ids = list(range(np_data.shape[0]))
    np.random.seed(11)
    np.random.shuffle(ids)

    np_data = np_data[ids]
    np_labels = np_labels[ids]

    return np_data, np_labels

if __name__ == "__main__":
    # prepare data
    np_data, np_labels = prepareData()
    batch_size = 4 #TODO: batchsize and seq_len is the issue to be addressed
    i = 5

    batches = batchify(np_data, batch_size, np_labels)
  
    import pdb; pdb.set_trace() 
    device = torch.device("cuda")     

    # setup model
    from model import RNN
    input_size = 1
    hidden_size = 3
    output_size = 2
    
    rnn = RNN(input_size, hidden_size, output_size)

    # define loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnn.parameters())
    
    
def train(rnn, input, mask, target, optimizer, criterion):
    loss_matrix = []    

    hidden = rnn.initHidden() 
    
    optimizer.zero_grad()
    
    input = input.view(-1,1,1) # seq_len X 1
    
    for t in range(input.size(0)):
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output, target.view(-1))
        loss_matrix.append(loss.view(1))

    loss_matrix = torch.cat(loss_matrix)
    loss = torch.sum(loss_matrix) / torch.sum(mask)

    loss.backward()
    
    optimizer.step()

    return output, loss.item()

print_every = 500
plot_every = 1000
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
n_batches = len(batches)

for i in range(0, n_batches):
    input, mask, target = get_batch(batches,i)
    output, loss = train(rnn, input, mask, target, optimizer, criterion)
    current_loss += loss

    # print iter number, loss, prediction, and target
    if i % print_every == (print_every - 1):
        top_n, top_i = output.topk(1)
        correct = 'correct' if top_i == target.item() else 'wrong'
        print("%d %d%% (%s) %.4f %d / %s" % (i, i / n_batches * 100, timeSince(start), loss, top_i, correct))

    if i % plot_every == (plot_every - 1):
        all_losses.append(current_loss / plot_every)
        current_loss=0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(all_losses)
plt.savefig("losses.png")

