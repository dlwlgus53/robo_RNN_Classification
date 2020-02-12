import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

from data import prepareData

def train(rnn, input, mask, target, optimizer, criterion):
    loss_matrix = []    

    hidden = rnn.initHidden().to(device)
    
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

def get_batch(inputs, masks, targets, bsz, i):
    return inputs[i:i+bsz],\
           masks[i:i+bsz],\
           targets[i:i+bsz]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    # prepare data
    device = torch.device("cpu")     
    
    inputs, masks, targets = prepareData()
    inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
       
 
    bsz = 1 #TODO: batchsize and seq_len is the issue to be addressed
    #i = 5
   

    # setup model
    from model import RNN
    input_size = 1
    hidden_size = 3
    output_size = 2
    
    rnn = RNN(input_size, hidden_size, output_size).to(device)

    # define loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnn.parameters())
    
    print_every = 500
    plot_every = 1000
    current_loss = 0
    all_losses = []

    start = time.time()
    n_batches = inputs.size(0)

    for i in range(0, n_batches, bsz):
        if (i+bsz) >= n_batches: continue
        input, mask, target = get_batch(inputs, masks, targets, bsz, i)

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

