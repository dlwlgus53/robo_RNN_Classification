import os
import random
import numpy as np
import argparse
import pandas as pd
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
class BatchDataset():
    def __init__(self, csv_path, batch_size, limit):
        self.batch_size = batch_size
        self.df_pd = pd.read_csv(csv_path, index_col =0 , header =0)#TODO change format
        self.df_np = self.remove_wrong_data(np.asarray(self.df_pd))
        self.limit = limit
        np.random.shuffle(self.df_np)    #TODO if test do not shuffle
        
        self.datalen = self.df_np.shape[0]
        self.n_batches = int(self.datalen / self.batch_size)-1
        
        self.report()


        #remove wrong data from raw data(empty data)
    def remove_wrong_data(self,data):
        cleans = []
        for n in range(data.shape[0]):
            if np.isnan(data[n,0:2]).any():
                continue
            cleans.append(data[n,:])
        return np.vstack(cleans)
        

        # return batchX, mask of X, batchY
    def getBatch(self,i):
            
        batchedX = []
        batchedY = []
        maskX = []

        start = i*self.batch_size
        end = (i+1)*self.batch_size

        batchedX = self.padding(self.df_np[start:end,:-1])
        batchedY = self.df_np[start:end,-1]

        maskX = self.masking(batchedX, batchedX.shape[0], batchedX.shape[1])
        
        # batchX, maskX, batchY
        return torch.from_numpy(np.vstack(batchedX)).type(torch.float32), torch.from_numpy(np.vstack(maskX)).type(torch.float32),\
             torch.from_numpy(np.vstack(batchedY)).type(torch.long)
       
        #padding x data    
    def padding(self,data):
        
        '''
            maxLen : longest data in group
            self.limit : limit of date to look
            length : shorter one between maxLen and self.limit
        '''


        maxLen = self.getMaxLen(data)-1 #  not to see last part
        length = maxLen if maxLen < self.limit else self.limit# set the day limit
        
        rows = []
        for n in range(data.shape[0]):
            row = []
            data_len = np.count_nonzero(~np.isnan(data[n,:]))
            short = 1 # 1 means, data[n] is shorter than length
            
            if (data_len>=length):
                short = 0
            
            if short:
                row = data[n,0:length-1]
                row = np.append(row,np.nan)
            else :
                row = data[n,0:length]

            if n == 0:
                rows = row 
            else :
                rows=np.vstack((rows,row))
        #print(rows)
        return np.vstack(rows)
                                                                                                                                                                      

    def getMaxLen(self,data):
        max =0
        for i in range(data.shape[0]):
            nonzero=np.count_nonzero(~np.isnan(data[i,:]))
            if(max<nonzero):
                max = nonzero
        return max
     
    # check the real part
    def masking(self,data, batch_size, maxlen):

        mask_rows = []


        for n in range(batch_size):
            mask_row = []
            throw = 0
            for i in range(maxlen):
                
                # other entries are nan
                if np.isnan(data[n,i]):
                    mask_row.append(0.)
                else:
                    # normal entry
                    mask_row.append(1)

            if throw == 0:
                mask_rows.append(mask_row)


        x = np.vstack(mask_rows)
        return x
            
            
    def report(self):
        print("shape of data")
        print(self.df_np.shape)
        print("number of batches")
        print(self.n_batches)


if __name__ == "__main__":
    data_path = "../data/norm"
    csv_path = os.path.join(data_path, "train.csv")
    
    sample = BatchDataset(csv_path, batch_size =5, limit = 10)
    X,X_mask, Y = sample.getBatch(3)
    print(X)
    print(X_mask)
    print(Y)
