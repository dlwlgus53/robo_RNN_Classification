import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

from data import prepareData, batchify

def get_batch(source, i):
    return torch.tensor(source[i][0]).type(torch.float32),\
           torch.tensor(source[i][1]).type(torch.float32),\
           torch.tensor(source[i][2]).type(torch.LongTensor)

def train(rnn, input, mask, target, optimizer, criterion):
    loss_matrix = []    

    hidden = rnn.initHidden()
    
    optimizer.zero_grad()
    
    input = input.unsqueeze(-1) # seq_len X 1
    
    for t in range(input.size(0) - 1):
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output, target.view(-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)
    mask = mask[:(input.size(0) - 1), :]
    
    masked = loss_matrix * mask
    
    loss = torch.sum(masked) / torch.sum(mask)

    loss.backward()
    
    optimizer.step()

    return output, loss.item()

def evaluate(rnn, input, mask, target, criterion):
    loss_matrix = []

    hidden = rnn.initHidden()

    input = input.unsqueeze(-1)
    
    for t in range(input.size(0) - 1):
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output, target.view(-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)
    mask = mask[:(input.size(0) - 1), :]
    
    masked = loss_matrix * mask
    
    loss = torch.sum(masked) / torch.sum(mask)

    return output, loss.item()

def validate(rnn, batches):
    current_loss = 0
    n_batches = len(batches)
    rnn.eval()
    with torch.no_grad(): 
        for i in range(0, n_batches):
            input, mask, target = get_batch(batches,i)
            output, loss = evaluate(rnn, input, mask, target, criterion)
            current_loss += loss
    
    return current_loss / n_batches

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    # prepare data
    np_data, np_labels, np_vdata, np_vlabels = prepareData()
    batch_size = 16 #TODO: batchsize and seq_len is the issue to be addressed
    n_epoches = 5000    

    batches = batchify(np_data, batch_size, np_labels)
    vbatches = batchify(np_vdata, batch_size, np_vlabels) 
 
    device = torch.device("cuda")     

    # setup model
    from model import RNN
    input_size = 1
    hidden_size = 3
    output_size = 2
    
    rnn = RNN(input_size, hidden_size, output_size, batch_size)

    # define loss
    criterion = nn.NLLLoss(reduction='none')
    optimizer = optim.RMSprop(rnn.parameters())
    
    print_every = 100
    current_loss = 0
    all_losses = []

    start = time.time()
    n_batches = len(batches)

    patience = 5    
    savePath = "./model.pth"
    
    for ei in range(n_epoches):
        bad_counter = 0
        best_loss = -1.0

        for i in range(0, n_batches):
            input, mask, target = get_batch(batches,i)
            output, loss = train(rnn, input, mask, target, optimizer, criterion)
            current_loss += loss

            # print iter number, loss, prediction, and target
            if i % print_every == (print_every - 1):
                top_n, top_i = output.topk(1)
                correct = 'correct' if top_i[0].item() == target[0].item() else 'wrong'
                print("%d %d%% (%s) %.4f %d / %s" % (i, i / n_batches * 100, timeSince(start), current_loss/print_every, top_i[0].item(), correct))

                current_loss=0

        valid_loss = validate(rnn, vbatches)
        all_losses.append(current_loss / print_every)
        
        if valid_loss < best_loss or best_loss < 0:
            bad_counter = 0
            torch.save(rnn, savePath)

        else:
            bad_counter += 1

        if bad_counter > patience:
            print('Early Stopping')
            break
         
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(all_losses)
    plt.savefig("losses.png")

