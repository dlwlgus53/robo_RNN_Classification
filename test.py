import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from model import RNN

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

loadPath = "./model.pth"
input_size = 1
hidden_size = 3
output_size = 2
batch_size = 1 # this is fixed to 1 at testing

#rnn = RNN(input_size, hidden_size, output_size, batch_size)
rnn = torch.load(loadPath)

df = pd.read_csv("classification_test.csv")
np_test = np.asarray(df)

np_data = np_test[:,:-1]
np_labels = np_test[:,-1].reshape(-1,1)

input, label = getSample(np_data, np_labels, 1)

hidden = torch.zeros(batch_size, hidden_size)

for t in range(input.size(0) - 1):
    output, hidden = rnn(input[t], hidden)
    
    logit, pred = output.topk(1)
    import pdb; pdb.set_trace()
