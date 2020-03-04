'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

import argparse

from data import FSIterator
from model import RNN
from train import train_main, test
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument('--logInterval', type=int, default=100, help='')
parser.add_argument('--saveModel', type=str, default = "bestmodel", help='')
parser.add_argument('--savePath', type=str, default = "png", help='')
parser.add_argument('--fileName', type=str, default="RMSprop0.001", help='')
parser.add_argument('--max_epochs', type=int, default=8, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--saveDir', type=str, default="png", help='')
parser.add_argument('--patience', type = int, default = 3, help='')
parser.add_argument('--daytolook', type = int, default = 7, help='')
parser.add_argument('--optim', type=str, default="RMSprop")# Adam, SGD, RMSprop
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate (no default)')

args = parser.parse_args()


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

    input_size = 2
    hidden_size = args.hidden_size
    output_size = 2
    
    model = RNN(input_size, hidden_size, output_size, batch_size).to(device)

    # define loss
    criterion = nn.NLLLoss(reduction='none')
    
    # define optimizer
    optimizer = "optim." + args.optim
    optimizer = eval(optimizer)(model.parameters(), lr=args.lr)
 
    logInterval = args.logInterval
    current_loss = 0
    all_losses = []

    start = time.time()

    patience = args.patience 
    savePath = args.savePath
   

    train_path = "../data/dummy/classification_train.csv"
    test_path = "../data/dummy/classification_test.csv"

    
    for ei in range(args.max_epochs):
        bad_counter = 0
        best_loss = -1.0
        
        train_main(args, model, train_path, criterion, optimizer)
        precision, f1, recallPerDays, accPerDays, valid_acc, lossPerDays, valid_loss = test(args, model, test_path, criterion)
        
        print("valid loss : {}".format(valid_loss))
        print(lossPerDays.tolist())
        print(accPerDays.tolist())
        print(valid_acc)
        print("recall")
        print(recallPerDays)
        print("f1")
        print(f1)
        print("precision")
        print(precision)
        if valid_loss < best_loss or best_loss < 0:
            print("find best")
            bad_counter = 0
            torch.save(model, args.saveModel)

        else:
            bad_counter += 1

        if bad_counter > patience:
            print('Early Stopping')
            break
         
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_path = os.path.join("./", args.saveDir) 
    plt.plot(accPerDays.tolist())
    plt.savefig(savePath + "/" + args.fileName + ".png")

