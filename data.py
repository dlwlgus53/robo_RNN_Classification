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

def addPadding(data):
    '''
    args: 2D npndarray with nans
    returns npndarray with nans padded with 0's
    '''
    for n in range(data.shape[0]):
        for i in range(data.shape[1]):
            if np.isnan(data[n,i]): data[n,i] = 0
    return data

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
        
        batch = addPadding(batch)

        batch = batch.transpose() # for inputting to RNN
        mask = mask.transpose()

        batches.append([batch, mask, target])

    return batches

def prepareData():
    df_train = pd.read_csv("classification_train.csv")
    df_valid = pd.read_csv("classification_valid.csv")

    np_train = np.asarray(df_train)
    np_valid = np.asarray(df_valid)
    
    np_data = np_train[:,:-1]
    np_labels = np_train[:,-1].reshape(-1,1)
    
    np_vdata = np_valid[:,:-1]
    np_vlabels = np_valid[:,-1].reshape(-1,1)

    return np_data, np_labels, np_vdata, np_vlabels

def get_batch(source, i):
    return torch.tensor(source[i][0]).type(torch.float32),\
           torch.tensor(source[i][1]).type(torch.float32),\
           torch.tensor(source[i][2]).type(torch.LongTensor)

if __name__ == "__main__":
    prepareData()

'''
if __name__ == "__main__":
    # prepare data
    np_data, np_labels = prepareData()
    batch_size = 16 #TODO: batchsize and seq_len is the issue to be addressed
    i = 5

    batches = batchify(np_data, batch_size, np_labels)
  
    device = torch.device("cuda")     

    # setup model
    from model import RNN
    input_size = 1
    hidden_size = 3
    output_size = 2
    
    rnn = RNN(input_size, hidden_size, output_size, batch_size)

    # define loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnn.parameters())
    
    
def train(rnn, input, mask, target, optimizer, criterion):
    loss_matrix = []    

    hidden = rnn.initHidden()
    
    optimizer.zero_grad()
    
    input = input.unsqueeze(-1) # seq_len X 1
    
    for t in range(input.size(0)):
        import pdb; pdb.set_trace()
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output, target.view(-1))
        loss_matrix.append(loss.view(1))

    loss_matrix = torch.cat(loss_matrix)
    loss = torch.sum(loss_matrix) / torch.sum(mask)

    loss.backward()
    
    optimizer.step()

    return output, loss.item()

print_every = 100
plot_every = 100
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
        correct = 'correct' if top_i[0].item() == target[0].item() else 'wrong'
        print("%d %d%% (%s) %.4f %d / %s" % (i, i / n_batches * 100, timeSince(start), loss, top_i[0].item(), correct))

    if i % plot_every == (plot_every - 1):
        all_losses.append(current_loss / plot_every)
        current_loss=0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(all_losses)
plt.savefig("losses.png")
'''
