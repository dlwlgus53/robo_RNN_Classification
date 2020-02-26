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

#from data import prepareData, batchify
from data2 import FSIterator
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--savePath', type=str, required=True, help='')
parser.add_argument('--max_epochs', type=int, default=2, help='')

args = parser.parse_args()

def train(rnn, input, mask, target, optimizer, criterion):
    rnn = rnn.train()
    loss_matrix = []
    #hidden = rnn.initHidden().to(device)
    #hidden = (hidden[0],hidden[1])
    
    optimizer.zero_grad()
    
    output, hidden = rnn(input)
    
    for t in range(input.size(0)):   
        loss = criterion(output[t].view(args.batch_size,-1), target.view(-1))
        loss_matrix.append(loss.view(1,-1))
    
    loss_matrix = torch.cat(loss_matrix, dim=0)
    
    masked = loss_matrix * mask
    
    loss = torch.sum(masked) / torch.sum(mask)

    loss.backward()
    
    optimizer.step()

    return loss.item()

def evaluate(rnn, input, mask, target, criterion):
    rnn = rnn.eval()
    loss_matrix = []
    
    output, hidden = rnn(input)

    for t in range(input.size(0)):
        loss = criterion(output[t].view(args.batch_size,-1), target.view(-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask

    lossPerDay = torch.sum(masked, dim = 1)/torch.sum(mask, dim=1 ) #1*daylen
    loss = torch.sum(masked) / torch.sum(mask)
    
    acc_matrix = []
    f1_matrix = []

    for t in range(input.size(0)):
        result = torch.max(output[t].data, 1)[1]
        accuracy = (target.squeeze() == result)
        acc_matrix.append((accuracy).view(1,-1))
    
    
    

    acc_matrix = torch.cat(acc_matrix, dim=0)

    masked_acc = acc_matrix * mask
    accPerDay = torch.sum(masked_acc, dim =1)/torch.sum(mask, dim=1)
    accuracy = torch.sum(masked_acc)/torch.sum(mask)
    
  
    return  accPerDay, accuracy.item(), lossPerDay, loss.item()

def validate(rnn, test_iter):
    current_loss = 0
    current_acc =0 
    lossPerDays = []
    accPerDays = []
    f1PerDays = []
    lossPerDays_avg = []
    accPerDays_avg = []
    f1PerDays_avg = []

    rnn.eval()
    with torch.no_grad():
        iloop =0  
        for input, target, mask, eof in test_iter:

            input = torch.tensor(input).type(torch.float32).to(device)
            target = torch.tensor(target).type(torch.LongTensor).to(device)
            mask = torch.tensor(mask).type(torch.float32).to(device)
 
            accPerDay, acc, lossPerDay, loss = evaluate(rnn, input, mask, target, criterion)
            lossPerDays.append(lossPerDay[:7]) #n_batches * 10
            accPerDays.append(accPerDay[:7])
            f1PerDays.append(f1PerDay[:7])
            current_acc += acc
            current_loss += loss
            iloop+=1

        lossPerDays = torch.stack(lossPerDays)
        lossPerDays_avg = lossPerDays.sum(dim =0)
        
        accPerDays = torch.stack(accPerDays)
        accPerDays_avg = accPerDays.sum(dim = 0)




            
        lossPerDays_avg = lossPerDays_avg/iloop
        accPerDays_avg = accPerDays_avg/iloop

        current_acc = current_acc/iloop
        current_loss = current_loss/iloop
    
    return  accPerDays_avg, current_acc, lossPerDays_avg, current_loss 

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    # prepare data
    batch_size = args.batch_size 
    n_epoches = args.max_epochs 

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

    patience = 5    
    savePath = args.savePath
   

    train_path = "../data/dummy/classification_train.csv"
    test_path = "../data/dummy/classification_test.csv"
    train_iter = FSIterator(train_path, batch_size)

    
    for ei in range(args.max_epochs):
        bad_counter = 0
        best_loss = -1.0
        
        iloop =0 
        for input,target, mask, eof in train_iter: #TODO for debugging
            input = torch.tensor(input).type(torch.float32).to(device)
            target = torch.tensor(target).type(torch.LongTensor).to(device)
            mask = torch.tensor(mask).type(torch.float32).to(device)
            rnn =rnn.to(device)
            loss = train(rnn, input, mask, target, optimizer, criterion)
            current_loss += loss

            if (iloop+1) % print_every == 0:
                #top_n, top_i = output.topk(1)
                #correct = 'correct' if top_i[0].item() == target[0].item() else 'wrong'
                # print minibatch, ongoing pecentage, time, currnet loss for minibatch
                print("%d  (%s) %.4f" % (iloop+1,timeSince(start), current_loss/print_every))
                all_losses.append(current_loss / print_every)

                current_loss=0
            
            iloop+=1


        
        test_iter = FSIterator(test_path, args.batch_size, 1)

        accPerDays, valid_acc, lossPerDays, valid_loss = validate(rnn, test_iter)
        print("valid loss : {}".format(valid_loss))
        print(lossPerDays)
        print(accPerDays)
        print(valid_acc)
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

