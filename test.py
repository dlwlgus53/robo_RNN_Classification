import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import math

def getIds(np_data, threshold):
    ids = []
    for n in range(np_data.shape[0]):
        for i in range(np_data.shape[1]):
            if np_data[n,i] > threshold:
                ids.append(n)
                break
                
    return ids

df = pd.read_csv("robo_train.csv")
np_data = np.asarray(df)

threshold = 3.0

ids = getIds(np_data, threshold)
idss = [idx for idx in list(range(np_data.shape[0])) if idx not in ids]

np_data = np_data[idss]

np.savetxt("robo_train2.csv", np_data, delimiter=',')

import pdb; pdb.set_trace()

'''
def cross_entropy(pred, soft_targets):
    return -soft_targets * torch.log(pred)

def L2(pred, soft_targets):
    return torch.pow(pred - soft_targets, 2)

labels = torch.tensor([[1,1,1,],[0,-1,-1],[1,1,-1]]).type(torch.float)
mask = torch.tensor([[1,1,1,],[1,0,0],[1,1,0]]).type(torch.float)

output1a = torch.tensor([[0.3,0.2,0.1],[0.5,0.3,0.3],[0.3,0.2,0.4]])
output1b = torch.tensor([[0.3,0.2,0.1],[0.5,0.9,0.9],[0.3,0.2,0.9]])

output2 = torch.tensor([[0.8,0.9,0.8],[0.4,0.9,0.9],[0.9,0.95,0.9]])

loss = L2

output1a = loss(output1a, labels)
output1b = loss(output1b, labels)

output1a = output1a * mask
output1b = output1b * mask

print(output1a)
print(output1b)

import pdb; pdb.set_trace()
'''