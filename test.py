import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import argparse

from model import RNN

parser = argparse.ArgumentParser()

parser.add_argument('--loadPath', type=str, required=True, help='')

args = parser.parse_args()

def getSeq_len(row):
    return np.count_nonzero(~np.isnan(row))

def getSample(np_data, np_labels, i):
    row = np_data[i]
    label = np_labels[i]
    
    row = row[:getSeq_len(row)]

    row, label = torch.tensor(row).type(torch.float32), torch.tensor(label).type(torch.LongTensor)
    row = row.view(-1, 1, 1)
    label = label.view(1)

    return row, label

loadPath = args.loadPath
batch_size = 1 # this is fixed to 1 at testing

#rnn = RNN(input_size, hidden_size, output_size, batch_size)
device = torch.device("cpu")

rnn = torch.load(loadPath).to(device)
#import pdb; pdb.set_trace()
hidden_size = rnn.state_dict()['rnn.weight_hh_l0'].shape[1]

df = pd.read_csv("classification_test.csv")
np_test = np.asarray(df)

np_data = np_test[:,:-1]
np_labels = np_test[:,-1].reshape(-1,1)

hidden = torch.zeros(2,1,batch_size, hidden_size)

n_totals = np.zeros(np_data.shape[1])
n_corrects = np.zeros(np_data.shape[1])

for n in range(np_data.shape[0]):
    input, label = getSample(np_data, np_labels, n)
    #input = input.unsqueeze(-1)

    for t in range(input.size(0) - 1):

        output, hidden = rnn(input[t], hidden)
    
        logit, pred = output.topk(1)
        n_totals[t] += 1

        if pred.item() == label.item():
            n_corrects[t] += 1

accs = []
for i in range(np_data.shape[1]):
    if n_corrects[i] != 0:
        accs.append(n_corrects[i] / n_totals[i])

mystring = loadPath

mylist = [str(num) for num in accs]

mystring = mystring + ',' + ','.join(mylist) + '\n'

with open("test.result", "a") as fp:
    fp.write(mystring)


