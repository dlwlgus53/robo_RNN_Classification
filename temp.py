import pandas as pd
import numpy as np

from data import addPadding

def addPadding1D(data):
    for i in range(data.shape[0]):
        if np.isnan(data[i]): data[i] = 0
    return data

df = pd.read_csv("./data/classification_valid.csv")

np_array = np.asarray(df)
np_data = np_array[:,:-1]
np_label = np_array[:,-1].reshape(-1,1)

col_prev = np_data[:,0].reshape(-1,1)
col = np_data[:,1].reshape(-1,1)

batch_in = np.array([[0,0.1,0.2,0.3],[0,0.1,0.1,0.2],[0,0.1,-0.1,-0.2]])

def getDelta(col_prev, col):
    delta = (col - col_prev).reshape(-1,1)

    return np.hstack([col, delta])

def addDelta(batch):
    timesteps = []
    # treat first column
    delta = np.zeros((batch.shape[0],2))
    timesteps.append(delta)

    # tread the rest
    for i in range(1, batch.shape[1]):
        timesteps.append(getDelta(batch[:,i-1].reshape(-1,1), batch[:,i].reshape(-1,1)))

    samples = np.stack(timesteps)

    return samples

batch_out = addDelta(batch_in)

import pdb; pdb.set_trace()


### add row by row ###
'''
row = np_data[0]
row = addPadding1D(row)

def addDelta(row):
    p_row = np.zeros(len(row))

    for i in range(1,len(row)):
        p_row[i] = row[i] - row[i-1]
  
    row = row.reshape(-1,1)
    p_row = p_row.reshape(-1,1)

    merged = np.hstack([row, p_row])

    return merged

new_sample = addDelta(row)

samples = []

for n in range(np_data.shape[0]):
    row = np_data[n]; row = addPadding1D(row)

    new_sample = addDelta(row)

    samples.append(new_sample)

np_new_data = np.stack(samples)

import pdb; pdb.set_trace() 
'''
