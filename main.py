'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

import argparse

from data import prepareData, batchify

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--savePath', type=str, required=True, help='')
parser.add_argument('--max_epochs', type=int, default=1, help='')

args = parser.parse_args()

def getBatch(source, i):
    return torch.tensor(source[i][0]).type(torch.float32).to(device),\
           torch.tensor(source[i][1]).type(torch.float32).to(device),\
           torch.tensor(source[i][2]).type(torch.LongTensor).to(device)

def train(rnn, input, mask, target, optimizer, criterion):

    loss_matrix = []
    #import pdb; pdb.set_trace()
    hidden = rnn.initHidden().to(device)
    hidden = (hidden[0],hidden[1])
    
    optimizer.zero_grad()
    
    #input = input.unsqueeze(-1) # depricated after addDelta()

    for t in range(input.size(0) - 1):
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output.view(args.batch_size,-1), target.view(-1))
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

    hidden = rnn.initHidden().to(device)

    #input = input.unsqueeze(-1) #deprecated after using addDelta()
    
    for t in range(input.size(0) - 1):
        output, hidden = rnn(input[t], hidden)
        loss = criterion(output.view(args.batch_size,-1), target.view(-1))
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
            input, mask, target = getBatch(batches,i)
            
            if (input.size(0)-1)==0: continue
            
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
    batch_size = args.batch_size #TODO: batchsize and seq_len is the issue to be addressed
    n_epoches = args.max_epochs 

    batches = batchify(np_data, batch_size, np_labels)
    vbatches = batchify(np_vdata, batch_size, np_vlabels) 
 
    device = torch.device("cuda")     

    # setup model
    from model import RNN, NaiveRNN
    input_size = 2
    hidden_size = args.hidden_size
    output_size = 2
    
    rnn = RNN(input_size, hidden_size, output_size, batch_size).to(device)
    #rnn = NaiveRNN(input_size, hidden_size, output_size, batch_size).to(device)

    # define loss
    criterion = nn.NLLLoss(reduction='none')
    optimizer = optim.RMSprop(rnn.parameters())
    
    print_every = 100 #print every minibatch
    current_loss = 0
    all_losses = []

    start = time.time()
    n_batches = len(batches)

    patience = 5    
    savePath = args.savePath
    
    for ei in range(args.max_epochs):
        bad_counter = 0
        best_loss = -1.0

        for i in range(0, n_batches): #TODO for debugging
            input, mask, target = getBatch(batches,i)

            if input.size(0) - 1 == 0: # single-day data
                continue

            output, loss = train(rnn, input, mask, target, optimizer, criterion)
            current_loss += loss

            # print iter number, loss, prediction, and target
            if (i+1) % print_every == 0:
                top_n, top_i = output.topk(1)
                #correct = 'correct' if top_i[0].item() == target[0].item() else 'wrong'
                # print minibatch, ongoing pecentage, time, currnet loss for minibatch
                print("%d %d%% (%s) %.4f" % (i+1, (i+1) / n_batches * 100, timeSince(start), current_loss/print_every))
                all_losses.append(current_loss / print_every)

                current_loss=0


        '''after one epoch, Test start'''

        valid_loss = validate(rnn, vbatches)
        print("valid loss : {}".format(valid_loss))
        
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
    plt.savefig(args.savePath + ".png")

