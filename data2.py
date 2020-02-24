import numpy as np

class FSIterator:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch
        self.fp = open(filename, 'r')

    def __iter__(self):
        return self

    def reset(self):
        self.fp.seek(0)

    def __next__(self):
        bat_seq = []
        end_of_data = 0
        for i in range(self.batch_size):
            seq = self.fp.readline()
            if seq == "":
                if self.just_epoch:
                    end_of_data = 1
                    if self.batch_size==1:
                        raise StopIteration
                    else:
                        break
                self.reset()
                seq = self.fp.readline() # read the first line

            seq_f = [float(s) for s in seq.split(',')]
            bat_seq.append(seq_f)

        x_data, y_data, mask_data = self.prepare_data(np.array(bat_seq))

        return x_data, y_data, mask_data, end_of_data

    def getSeq_len(self,row):
        '''                                                                                                                                 
        returns: count of non-nans (integer)
        adopted from: M4rtni's answer in stackexchange
        '''
        return np.count_nonzero(~np.isnan(row))


    def getMask(self,batch):
        '''
        returns: boolean array indicating whether nans
        '''
        return (~np.isnan(batch)).astype(np.int32)

    def trimBatch(self,batch):
        '''
        args: npndarray of a batch (bsz, n_features)
        returns: trimmed npndarray of a batch.
        '''
        max_seq_len = 0
        for n in range(batch.shape[0]):
            max_seq_len = max(max_seq_len, self.getSeq_len(batch[n]))

        if max_seq_len == 0:
            print("error in trimBatch()")
            sys.exit(-1)

        batch = batch[:,:max_seq_len]
        return batch

    
    def prepare_data(self, seq):
        PRE_STEP = 1 # this is for delta
        #iimport pdb; pdb.set_trace()
        seq_x = seq[:,:-1]
        seq_y = seq[:,-1]
        
        seq_x = self.trimBatch(seq_x)
        seq_mask = self.getMask(seq_x[:,1:-PRE_STEP])
        seq_x = np.nan_to_num(seq_x)

        seq_x_delta = seq_x[:,1:] - seq_x[:,:-1]


        x_data = np.stack([seq_x[:,1:-PRE_STEP], seq_x_delta[:,:-PRE_STEP] ], axis=2)#batch * daylen * inputdim(2)
        x_data = x_data.transpose(1,0,2)# daylen * batch * inputdim
        
        y_data = seq_y.reshape(1,-1) # batch * 1
        y_data = np.stack([y_data.transpose(1,0)])# 1*batch*1

        #y_data = (seq_delta[:,1:] > 0)*1.0 # the diff
        
        mask_data = np.stack([seq_mask.transpose(1,0)])
        '''
        x_data : daymaxlen-2, batch, inputdim(=2)
        y_data : 1 * batch * 1
        mask_data : 1*daymaxlen-2, batch
        '''
        return x_data, y_data, mask_data

if __name__ == "__main__":
    import os
    import numpy as np

    #filename = os.environ['HOME']+'/FinSet/data/GM.csv.seq.shuf'
    filename = "../data/dummy/classification_train.csv"
    #df_train = pd.read_csv("../data/dummy/classification_train.csv")
    bs = 4
    train_iter = FSIterator(filename, batch_size=bs, just_epoch=True)

    i = 0
    for tr_x, tr_y, tr_m, end_of_data in train_iter:
        print(i, tr_x, tr_y, tr_m)
        i = i + 1
        if i > 2:
            break
                    
