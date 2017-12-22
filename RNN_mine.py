import numpy as np
from six.moves import xrange
import os
import platform
import time
def softmax(x):
    #orig_shape = x.shape
    #if len(x.shape)>1:
    #    s = np.max(x,axis=1)
    #    s = s[:,np.newaxis] # necessary stop to do broadcasting
    #    e_x = np.exp(x-s)
    #    div = np.sum(e_x,axis=1)
    #    div = div[:, np.newaxis]
    #    return e_x/div
    #else:
    #    e_x = np.exp(x-np.max(x))
    #    x = e_x/e_x.sum()
    #    return x
    return np.exp(x)/np.sum(np.exp(x))
class RNNModel:
    def __init__(self):
        # hyper parameter
        self.hidden_size = 100
        self.learning_rate = 1e-1
        self.seq_length = 25
        self.data_size = 0
        self.vocab_size = 0
    def load_data(self):
        start = time.time()
        if platform.system() == 'Linux':
            filename = os.getcwd() + '/Data/LoR_full'
        else:
            filename = os.getcwd() + '\\Data\\LoR_full'
        # read data
        data = open(filename,'r',encoding='utf-8').read()
        # get characters
        chars = list(set(data))
        self.vocab_size, self.data_size = len(chars), len(data)
        # create dictionary of characters
        char_to_idx = {ch:i for i,ch in enumerate(chars)}
        idx_to_char = {i:ch for i,ch in enumerate(chars)}
        print('Time to load data {:.2f}'.format(time.time() - start))
        return data, char_to_idx, idx_to_char 
    def create_weights(self):
        hs = self.hidden_size
        vs = self.vocab_size

        Wx = np.random.randn(hs,vs) * 0.01
        Wh = np.random.randn(hs,hs) * 0.01
        Wy = np.random.randn(vs,hs) * 0.01
        bh = np.zeros((hs,1))
        by = np.zeros((vs,1))
        return Wx, Wh, Wy, bh, by
    def generate_batch(self,index,data,char_to_idx):
        idx = index
        if idx+1+self.seq_length > self.data_size:
            idx = 0
        inputs = [char_to_idx[ch] for ch in data[idx:idx+self.seq_length]]
        labels = [char_to_idx[ch] for ch in
                data[idx+1:idx+1+self.seq_length]]
        return idx+1, inputs, labels
    def feed_forward(self,x,hprev,Wx,Wh,Wy,bh,by):
        hnext = np.tanh(np.dot(Wx,x) + np.dot(Wh,hprev) + bh)
        y_hat = softmax(np.dot(Wy,hnext) + by)
        return y_hat, hnext
    def loss_func(self, y, y_hat):
        #print(y)
        return -1 * (np.dot(y.T, np.log(y_hat)) + np.dot((1-y).T, np.log(1-y_hat)))
    def backprop (self,x, y, y_hat, loss,hprev, hnext, Wx, Wh, Wy, bh, by):
        dhnext = np.zeros_like(hnext)
        dy = np.copy(y_hat)
        dy[y] -= 1
        dWy = np.dot(dy, hnext.T)
        # output bias
        dby = dy
        dh = np.dot(Wy.T,dy) + dhnext
        dhraw = (1-hnext*hnext)*dh
        dbh = dhraw #derivative of hidden_size
        dWx = np.dot(dhraw,x.T)
        dWh = np.dot(dhraw,hprev.T) #derivative of hidden layer to hidden layer weight
        dhnext = np.dot(Wh.T, dhraw)
        return dWx, dWh, dWy, dbh, dby 
if __name__ == "__main__":
    model = RNNModel()
    data, char_to_idx, idx_to_char = model.load_data()
    Wx, Wh, Wy, bh,by = model.create_weights()
    
    hprev = np.zeros((model.hidden_size,1))
    x, h, y_hat, ps = {}, {}, {}, {}
    loss = 0
    for index in xrange(model.seq_length * 100):
        idx, inputs, labels = model.generate_batch(index, data,char_to_idx)
        for i in xrange(len(inputs)):
            x[i] = np.zeros((model.vocab_size,1))
            x[i][inputs[i]]  = 1
            y_hat[i], h[i] = model.feed_forward(x[i],hprev,Wx,Wh,Wy,bh,by)
            target = np.zeros((model.vocab_size,1))
            target[labels[i]] = 1
            loss += model.loss_func(target, y_hat[i])
            #print (loss)
        dWx, dWh, dWy, dbh, dby = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy), np.zeros_like(bh), np.zeros_like(by)
        for i in reversed(xrange(1,len(inputs))):
            dWx_i, dWh_i, dWy_i, dbh_i, dby_i = model.backprop(x[i],labels[i],y_hat[i],loss, h[i-1], h[i], Wx, Wh, Wy, bh, by)
            dWx += dWx_i
            dWh += dWh_i
            dWy += dWy_i
            dbh += dbh_i
            dby += dby_i
        for dparam in [dWx, dWh, dWy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) #clip to avoid gradient exploding
		



