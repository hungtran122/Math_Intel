import numpy as np
from six.moves import xrange
import platform
import os
import time
if platform.system() == 'Linux':
	path = os.getcwd() + '/Data/LoR_full'
else:
	path = os.getcwd() + '\\Data\\LoR_full'

# read data
start = time.time()
data = open(path,'r',encoding = 'utf-8').read() #all character in corpus
print('It tooks {:.10f} second to read data'.format(time.time() - start))
chars = list(set(data)) #characters appear in data
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
print('Size of data = {:d} and number of chars {:d}'.format(data_size, vocab_size))

vector_for_char_a = np.zeros((vocab_size,1))
vector_for_char_a[char_to_ix['a']] = 1
#print(vector_for_char_a.ravel())

#hyperparameter
hidden_size = 100
seq_length = 25
learning_rate = 1e-1
Wx = np.random.randn(hidden_size,vocab_size) * 0.01
Wh = np.random.randn(hidden_size,hidden_size) * 0.01
Wy = np.random.randn(vocab_size,hidden_size) * 0.01
bh = np.zeros((hidden_size,1))
by = np.zeros((vocab_size,1))
def lossFun(inputs,targets,hprev):

	#store our inputs, hidden states, outputs, and probability values
	xs, hs, ys, ps = {}, {}, {}, {}
	# we don't want hs[-1] to automatically change if hprev is changed
	hs[-1] = np.copy(hprev)
	#init loss
	loss = 0
	#feedforward
	for t in xrange(len(inputs)):
		xs[t] = np.zeros((vocab_size,1))
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wx,xs[t]) + np.dot(Wh,hs[t-1]) + bh)
		ys[t] = np.dot(Wy,hs[t]) + by
		ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t])) # probabilities for next chars
		#print(ps)
		loss += -np.log(ps[t][targets[t],0]) # cross entropy loss
	#backward
	dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	for t in reversed(xrange(len(inputs))):
		#output probabilities
		dy = np.copy(ps[t])
		#derive our first gradient
		dy[targets[t]] -= 1 #backprop into y
		dWy += np.dot(dy, hs[t].T)
		#derivative of output bias
		dby += dy
		#backpropagate
		dh = np.dot(Wy.T,dy) + dhnext #backprop into h
		dhraw = (1-hs[t] * hs[t]) * dh
		dbh += dhraw #derivative of hidden bias
		dWx += np.dot(dhraw,xs[t].T) #derivative of input to hidden layer weight
		dWh += np.dot(dhraw, hs[t-1].T) #derivative of hidden layer to hidden layer weight
		dhnext = np.dot(Wh.T, dhraw)
	for dparam in [dWx, dWh, dWy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam) #clip to mitigate exploding gradients
	return loss, dWh, dWx, dWy, dbh, dby, hs[len(inputs) -1]

#prediction, one full forward pass
def sample(h, seed_ix, n):
	"""                                                                                                                                                                                         
	sample a sequence of integers from the model                                                                                                                                                
	h is memory state, seed_ix is seed letter for first time step   
	n is how many characters to predict
	"""
	#create vector
	x = np.zeros((vocab_size, 1))
	#customize it for our seed char
	x[seed_ix] = 1
	#list to store generated chars
	ixes = []
	#for as many characters as we want to generate
	for t in xrange(n):
		#a hidden state at a given time step is a function 
		#of the input at the same time step modified by a weight matrix 
		#added to the hidden state of the previous time step 
		#multiplied by its own hidden state to hidden state matrix.
		h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + bh)
		#compute output (unnormalised)
		y = np.dot(Wy, h) + by
		## probabilities for next chars
		p = np.exp(y) / np.sum(np.exp(y))
		#pick one with the highest probability 
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		#create a vector
		x = np.zeros((vocab_size, 1))
		#customize it for the predicted char
		x[ix] = 1
		#add it to the list
		ixes.append(ix)

	txt = ''.join(ix_to_char[ix] for ix in ixes)
	print ('----\n %s \n----' % (txt, ))
hprev = np.zeros((hidden_size,1)) # reset RNN memory  
#predict the 200 next characters given 'a'
sample(hprev,char_to_ix['a'],200)

n,p = 0,0
mWx, mWh, mWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size) * seq_length #loss at iteration 0
start = time.time()
while n< 10000*10:
	if p+seq_length+1>=len(data) or n==0:
		hprev = np.zeros((hidden_size,1)) #reset RNN memory
		p = 0 #go from start of data
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	#forward seq_length characters through the net and fetch gradient
	loss, dWh, dWx, dWy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
	smooth_loss = smooth_loss * 0.999 + loss * 0.001
	

	if n % 10000 == 0:
		print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
		sample(hprev, inputs[0], 200)

	# perform parameter update with Adagrad                                                                                                                                                     
	for param, dparam, mem in zip([Wx, Wh, Wy, bh, by],
	                            [dWx, dWh, dWy, dbh, dby],
	                            [mWx, mWh, mWy, mbh, mby]):
		
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update                                                                                                                   

	p += seq_length # move data pointer                                                                                                                                                         
	n += 1 # iteration counter
print ("Traning time {:.2f}".format(time.time() - start))