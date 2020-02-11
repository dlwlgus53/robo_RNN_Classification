import pandas as pd
import numpy as np
import math
import sys

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

        batches.append([batch, mask, target])

    return batches

df_train = pd.read_csv("robo_train.csv")
df_dummy = pd.read_csv("robo_dummy.csv")

np_train = np.asarray(df_train)
np_dummy = np.asarray(df_dummy)

ones = np.ones((np_train.shape[0],1))
zeros = np.zeros((np_dummy.shape[0],1))

# concatconcat
np_data = np.vstack([np_train, np_dummy])
np_labels = np.vstack([ones, zeros])

# shuffle
ids = list(range(np_data.shape[0]))
np.random.seed(11)
np.random.shuffle(ids)

np_data = np_data[ids]
np_labels = np_labels[ids]

# batchify
batches = batchify(np_data, 4, np_labels)
import pdb; pdb.set_trace()
