import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

def getSeq_len(row):
    '''
    returns: count of non-nans (integer)
    adopted from: M4rtni's answer in stackexchange
    '''
    return np.count_nonzero(~np.isnan(row))

def getMask(batch):
    '''
    returns: boolean array indicating whether nans
    '''
    return (~np.isnan(batch)).astype(np.int32)

def trim_batch(batch):
    '''
    args: npndarray of a batch (bsz, n_features)
    returns: trimmed npndarray of a batch.
    '''
    max_seq_len = 0
    for n in range(batch.shape[0]):
        max_seq_len = max(max_seq_len, getSeq_len(batch[n]))

    if max_seq_len == 0:
        print("error in trim_batch()")
        sys.exit(-1)

    batch = batch[:,:max_seq_len]
    return batch

def addPadding(data):
    '''
    args: 2D npndarray with nans
    returns npndarray with nans padded with 0's
    '''
    for n in range(data.shape[0]):
        for i in range(data.shape[1]):
            if np.isnan(data[n,i]): data[n,i] = 0
    return data

def batchify(data, bsz, labels):
    batches = []
    
    n_samples = data.shape[0]
    for n in range(0,n_samples,bsz):
        if n+bsz > n_samples: #discard remainder #TODO: use remainders somehow
            break
        batch = data[n:n+bsz]
        target = labels[n:n+bsz]

        batch = trim_batch(batch)
        mask = getMask(batch)
        
        batch = addPadding(batch)

        batch = batch.transpose() # for inputting to RNN
        mask = mask.transpose()

        batches.append([batch, mask, target])

    return batches

def prepareData():
    df_train = pd.read_csv("classification_train.csv")
    df_valid = pd.read_csv("classification_valid.csv")

    np_train = np.asarray(df_train)
    np_valid = np.asarray(df_valid)
    
    np_data = np_train[:,:-1]
    np_labels = np_train[:,-1].reshape(-1,1)
    
    np_vdata = np_valid[:,:-1]
    np_vlabels = np_valid[:,-1].reshape(-1,1)

    return np_data, np_labels, np_vdata, np_vlabels

