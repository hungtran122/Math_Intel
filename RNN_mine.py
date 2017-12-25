import numpy as np
from six.moves import xrange
import os
import platform
import time
import matplotlib.pyplot as plt
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
        self.hidden_size = 300
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
        return inputs, labels
    def feed_forward(self,x,hprev,Wx,Wh,Wy,bh,by):
        hnext = np.tanh(np.dot(Wx,x) + np.dot(Wh,hprev) + bh)
        #y_hat = softmax(np.dot(Wy,hnext) + by)
        ys = np.dot(Wy,hnext) + by
        y_hat = np.exp(ys)/np.sum(np.exp(ys))
        return y_hat, hnext
    def loss_func(self, target, y_hat):
        loss = -np.log(y_hat[target,0])
        return loss
    def backprop (self,inputs,targets,x,y_hat,h,Wx,Wh,Wy,bh,by):
        dhnext = np.zeros_like(h[0])
        dWx, dWh, dWy, dbh, dby = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy), np.zeros_like(bh), np.zeros_like(by)
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(y_hat[t])
            dy[targets[t]] -= 1
            dWy += np.dot(dy, h[t].T)
            # output bias
            dby += dy
            dh = np.dot(Wy.T,dy) + dhnext
            dhraw = (1-h[t]*h[t])*dh
            dbh += dhraw #derivative of hidden_size
            dWx += np.dot(dhraw,x[t].T)
            dWh += np.dot(dhraw,h[t-1].T) #derivative of hidden layer to hidden layer weight
            dhnext = np.dot(Wh.T, dhraw)
        for dparam in [dWx, dWh, dWy, dbh, dby]:
            np.clip(dparam, -5 , 5, out=dparam) # avoid exploiding gradients
        return dWx, dWh, dWy, dbh, dby, h[len(inputs) - 1] 
    def sample(self,h,seed_ix,n,idx_to_char,Wx,Wh,Wy,bh,by):
        #create vector
        x = np.zeros((self.vocab_size,1))
        x[seed_ix] = 1
        #list to store generated chars
        ixes = []
        #for as many characters as we want to generate
        for t in xrange(n):
            h = np.tanh(np.dot(Wx,x) + np.dot(Wh,h) + bh)
            y = np.dot(Wy,h) + by
            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size),p=p.ravel())
            x = np.zeros((self.vocab_size,1))
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(idx_to_char[ix] for ix in ixes)
        print ('---\n %s \n ---' %(txt,))
if __name__ == "__main__":
    model = RNNModel()
    data, char_to_idx, idx_to_char = model.load_data()
    Wx, Wh, Wy, bh,by = model.create_weights()
    mWx, mWh, mWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    smooth_loss = -np.log(1.0/model.vocab_size) * model.seq_length
    idx = 0
    print('Length data = %d, Sequence length = %d' %(len(data), model.seq_length))
    hprev = np.zeros((model.hidden_size,1))
    x_plt, y_plt = [], []
    start = time.time()
    for index in xrange(10000*200):
        x, h, y_hat= {}, {}, {}
        h[-1] = np.copy(hprev)
        if idx + 1 + model.seq_length >= len(data) or index == 0: 
            idx = 0
            hprev = np.zeros((model.hidden_size,1))
        loss = 0
        inputs, labels = model.generate_batch(idx, data,char_to_idx)
        for t in xrange(len(inputs)):
            x[t] = np.zeros((model.vocab_size,1))
            x[t][inputs[t]]  = 1
            y_hat[t], h[t] = model.feed_forward(x[t],h[t-1],Wx,Wh,Wy,bh,by)
            #target = np.zeros((model.vocab_size,1))
            #target[labels[t]] = 1
            loss += model.loss_func(labels[t],y_hat[t])
        smooth_loss = 0.999 * smooth_loss + loss * 0.001
        if smooth_loss < 40:
            print ("Break because smooth_loss = {:.5f} and it's less than 40".format(smooth_loss))
            break;
        if (index % 10000) == 0:
            print ("iter: {:}, loss: {:}".format(index,smooth_loss))
            x_plt.append(smooth_loss)
            y_plt.append(index)
            model.sample(hprev,inputs[0],200,idx_to_char,Wx,Wh,Wy,bh,by)
        dWx, dWh, dWy, dbh, dby, hprev = model.backprop(inputs, labels, x, y_hat, h, Wx, Wh, Wy, bh, by)
        for param,dparam, mem in zip ([Wx, Wh, Wy, bh, by],\
                [dWx, dWh, dWy, dbh, dby],\
                [mWx, mWh, mWy, mbh, mby]):
            mem += dparam * dparam
            param += -model.learning_rate * dparam/np.sqrt(mem + 1e-8) #adagrad update
        idx += model.seq_length # move pointer 
    plt.plot(y_plt, x_plt)
    plt.show()
    print ('Total time for training is {:.2f}'.format(time.time() - start))
