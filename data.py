import numpy as np
import torch

class FSIterator:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch        
        self.fp_end = open(filename + "/end.csv", 'r')
        self.fp_start = open(filename + "/start.csv", 'r')
        self.fp_low = open(filename + "/low.csv", 'r')
        self.fp_high = open(filename + "/high.csv", 'r')
        self.fp_trade = open(filename + "/trade.csv", 'r')
        self.fps = [self.fp_end, self.fp_start, self.fp_low, self.fp_high,self.fp_trade]

    def __iter__(self):
        return self

    def reset(self):
        for fp in self.fps:
            fp.seek(0)

    def __next__(self):
 
        bat_seq = []
        touch_end = 0

        while(len(bat_seq)< self.batch_size):
            seq_end = self.fps[0].readline()
            seq_start = self.fps[1].readline()
            seq_low = self.fps[2].readline()
            seq_high = self.fps[3].readline()
            seq_trade = self.fps[4].readline()
                
            if touch_end:
                raise StopIteration

            if seq_end == "":
                print("touch end")
                touch_end = 1

                '''
                if self.just_epoch:
                    end_of_data = 1
                    if self.batch_size==1:
                        raise StopIteration
                    else:
                        break
                '''
                self.reset()
                # read the first line
                seq_end = self.fps[0].readline()
                seq_start = self.fps[1].readline()
                seq_low = self.fps[2].readline()
                seq_high = self.fps[3].readline()
                seq_trade = self.fps[4].readline()

            seq_end = [float(s) for s in seq_end.split(',')]
            seq_start = [float(s) for s in seq_start.split(',')]
            seq_low = [float(s) for s in seq_low.split(',')]
            seq_high = [float(s) for s in seq_high.split(',')]
            seq_trade = [float(s) for s in seq_trade.split(',')]

            #if(np.count_nonzero(~np.isnan(seq_end))>=21):
                # and np.count_nonzero(~np.isnan(seq_end))< 21*/):
                #if(np.count_nonzero(~np.isnan(seq_f))>4):
            if(sum(~np.isnan(seq_end)) == sum(~np.isnan(seq_start)) == sum(~np.isnan(seq_low))== sum(~np.isnan(seq_high)) == sum(~np.isnan(seq_trade))):
                if(max(seq_end)<=2):
                    seqs = [seq_end, seq_start, seq_low, seq_high, seq_trade]
                    bat_seq.append(seqs)
                
        x_data, y_data, mask_data = self.prepare_data(np.array(bat_seq))#B x [[E*daylen],[S*daylen],[L*daylen],[H*daylen]]
        
        device = torch.device("cuda")
        x_data = torch.tensor(x_data).type(torch.float32).to(device)
        y_data = torch.tensor(y_data).type(torch.LongTensor).to(device)
        mask_data = torch.tensor(mask_data).type(torch.float32).to(device)

        return x_data, y_data, mask_data

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

    def trimBatch(self, batch):
        '''
        args: npndarray of a batch (bsz, n_features)
        returns: trimmed npndarray of a batch.
        '''
        max_seq_len = 0
        for n in range(batch.shape[0]):
            batch[n,self.getSeq_len(batch[n])-1] = np.nan

        for n in range(batch.shape[0]):
            max_seq_len = max(max_seq_len, self.getSeq_len(batch[n]))
        
        if max_seq_len == 0:
            print("error in trimBatch()")
            sys.exit(-1)

        batch = batch[:,:max_seq_len]
        return batch

    
    def prepare_data(self, seq):
        PRE_STEP = 1 # this is for delta
        #import pdb; pdb.set_trace()
        seq_end_x = seq[:,0,:-1]
        seq_start_x = seq[:,1,:-1]
        seq_low_x = seq[:,2,:-1]
        seq_high_x = seq[:,3,:-1]
        seq_trade_x = seq[:,4,:-1]
        
        seq_y = seq[:,0,-1]
        
        # resize into the longest day length
        seq_end_x = self.trimBatch(seq_end_x)
        seq_start_x = self.trimBatch(seq_start_x)
        seq_low_x = self.trimBatch(seq_low_x)
        seq_high_x = self.trimBatch(seq_high_x)
        seq_trade_x = self.trimBatch(seq_trade_x)
        
        seq_mask = self.getMask(seq_end_x[:,1:-PRE_STEP])
        
        seq_end_x = np.nan_to_num(seq_end_x)
        seq_start_x = np.nan_to_num(seq_start_x)
        seq_low_x = np.nan_to_num(seq_low_x)
        seq_high_x = np.nan_to_num(seq_high_x)
        seq_trade_x = np.nan_to_num(seq_trade_x)

        seq_end_x_delta = seq_end_x[:,1:] - seq_end_x[:,:-1]
        seq_start_x_delta = seq_start_x[:,1:] - seq_start_x[:,:-1]
        seq_low_x_delta = seq_low_x[:,1:] - seq_low_x[:,:-1]
        seq_high_x_delta = seq_high_x[:,1:] - seq_high_x[:,:-1]
        seq_trade_x_delta = seq_trade_x[:,1:] - seq_trade_x[:,:-1]
        try : 
            x_data = np.stack([seq_end_x[:,1:-PRE_STEP], seq_end_x_delta[:,:-PRE_STEP],
                           seq_start_x[:,1:-PRE_STEP], seq_start_x_delta[:,:-PRE_STEP],
                           seq_low_x[:,1:-PRE_STEP], seq_low_x_delta[:,:-PRE_STEP],
                           seq_high_x[:,1:-PRE_STEP], seq_high_x_delta[:,:-PRE_STEP],
                           seq_trade_x[:,1:-PRE_STEP], seq_trade_x_delta[:,:-PRE_STEP]], axis=2) #batch * daylen * inputdim
        except:
            import pdb; pdb.set_trace()
        x_data = x_data.transpose(1,0,2) # daylen * batch * inputdim
        
        y_data = seq_y.reshape(1,-1) # batch * 1
        y_data = np.stack([y_data.transpose(1,0)])# 1*batch*1

        #y_data = (seq_delta[:,1:] > 0)*1.0 # the diff
        
        mask_data = np.stack(seq_mask.transpose(1,0))
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
                    
