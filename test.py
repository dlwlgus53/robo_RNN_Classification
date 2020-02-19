import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import argparse

from model import RNN

parser = argparse.ArgumentParser()

parser.add_argument('--loadPath', type=str, required=True, help='')

args = parser.parse_args()

def addDelta(row):
    p_row = np.zeros(len(row))

    for i in range(1,len(row)):
        p_row[i] = row[i] - row[i-1]
  
    row = row.reshape(-1,1)
    p_row = p_row.reshape(-1,1)

    merged = np.hstack([row, p_row])

    return merged

def getSeq_len(row):
    return np.count_nonzero(~np.isnan(row))

def getSample(np_data, np_labels, i):
    row = np_data[i]
    label = np_labels[i]
    
    row = row[:getSeq_len(row)]

    sample = addDelta(row)
    #import pdb; pdb.set_trace()

    sample, label = torch.tensor(sample).type(torch.float32), torch.tensor(label).type(torch.LongTensor)
    sample = sample.view(-1, 1, 2)
    label = label.view(1)

    return sample, label

loadPath = args.loadPath
batch_size = 1 # this is fixed to 1 at testing

#rnn = RNN(input_size, hidden_size, output_size, batch_size)
device = torch.device("cpu")

rnn = torch.load(loadPath).to(device)

if (str(rnn).split('('))[0] == 'NaiveRNN':
    hidden_size = rnn.state_dict()['i2h.weight'].shape[0]
    hidden = torch.zeros(1,hidden_size)
else:
    hidden_size = rnn.state_dict()['rnn.weight_hh_l0'].shape[1]
    hidden = torch.zeros((2,1,1,hidden_size))


df = pd.read_csv("./data/classification_test.csv")
np_test = np.asarray(df)

np_data = np_test[:,:-1]
np_labels = np_test[:,-1].reshape(-1,1)

n_totals = np.zeros(np_data.shape[1])
#n_corrects = np.zeros(np_data.shape[1])

n_preds = np.zeros((2, np_data.shape[1]))
n_targets = np.zeros((2, np_data.shape[1]))
n_corrects = np.zeros((2, np_data.shape[1]))

for n in range(np_data.shape[0]):
    print(n)
    
    input, label = getSample(np_data, np_labels, n)
    #input = input.unsqueeze(-1)

    for t in range(input.size(0) - 1):

        output, hidden = rnn(input[t], hidden)
    
        logit, pred = output.topk(1)
       
        n_totals[t] += 1
        n_preds[pred.item(),t] += 1
        n_targets[label.item(),t] += 1
        
        if pred.item() == label.item():
            n_corrects[pred.item(),t] += 1

accs = []
precs = []
recs = []

for i in range(np_data.shape[1]):
    if n_totals[i] != 0:
        accs.append(np.sum(n_corrects[:,i]) / n_totals[i])

for i in range(np_data.shape[1]):
    if n_preds[1, i] != 0:
        precs.append(n_corrects[1,i] / n_preds[1,i])
    else:
        precs.append(-1.0)

for i in range(np_data.shape[1]):
    if n_targets[1, i] != 0:
        recs.append(n_corrects[1,i] / n_targets[1,i])
    else:
        recs.append(-1.0)
 
mystring = loadPath

acclist = [str(num) for num in accs]
preclist = [str(num) for num in precs]
reclist = [str(num) for num in recs]

accString = mystring + ',' + 'accuracy' + ',' + ','.join(acclist) + '\n'
precString = mystring + ',' + 'precision' + ',' + ','.join(preclist) + '\n'
recString = mystring + ',' + 'recall' + ',' + ','.join(reclist) + '\n'

with open("test.result", "a") as fp:
    fp.write(accString)
    fp.write(precString)
    fp.write(recString)
