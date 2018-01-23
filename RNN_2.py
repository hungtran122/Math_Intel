import sys
import operator
import nltk
import os
import platform
import time
import numpy as np

filename = 'MyText.txt'
if platform.system == 'Linux':
	filepath = os.getcwd() + '/Data/'
else:
	filepath = os.getcwd() + '\\Data\\'
start = time.time()
print ('Start reading data ...')
vocab_size = 80
start_token = "START_SENT"
end_token = "END_SENT"
unknown_token = "UNKNOWN_TOKEN"
# Step 1: Reading data from file
# using nltk to tokenize text

text = nltk.corpus.gutenberg.raw(filepath + filename)
sents = nltk.sent_tokenize(text)
sentences = ['%s %s %s' %(start_token, x , end_token + " ") for x in sents]
sentences_str = ''.join(sentences)
words = nltk.word_tokenize(sentences_str)
word_fred = nltk.FreqDist(words)

vocab = word_fred.most_common(vocab_size - 1)

print ('The corpus has {:} sentences.\nAnd 10th sentence is:\n{:}'.format(len(sentences), sentences[9]))
print ('The corpus has {:} unique words'.format(len(word_fred.items())))
print ("10 most common words is:\n{:}".format(vocab[:10]))
print ('It took {:.2f} seconds to process data'.format(time.time() - start))

idx_to_word = [w for w in vocab]
idx_to_word.append(unknown_token)
word_to_idx = {w:i for i,w in enumerate(idx_to_word)}
#word_to_idx = dict([w,i] for i,w in enumerate(idx_to_word))

start = time.time()
# replace words with unknown_token
for i, sentence in enumerate(sentences):
	sentences[i] = [w if w in word_to_idx else unknown_token for w in sentence]
print('Replacing words with unknown_token took {:.2f} seconds'.format(time.time() - start))

# Step 2: create training data
X_train = np.asarray([[word_to_idx[w] for w in sentence[:-1]] for sentence in sentences])
y_train = np.asarray([[word_to_idx[w] for w in sentence[1:]] for sentence in sentences])

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)
class RNN_numpy:
	def __init__(self, hidden_dim, learning_rate = 0.005, bptt_truncate = 4):
		self.word_dim = vocab_size
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		self.learning_rate = learning_rate
		self.createWeights()
	def createWeights(self):
		bound_1 = np.sqrt(1/self.word_dim)
		bound_2 = np.sqrt(1/self.hidden_dim)

		self.Wx = np.random.uniform(-bound_1, bound_1, (self.hidden_dim, self.word_dim))
		self.Wh = np.random.uniform(-bound_2, bound_2, (self.hidden_dim, self.hidden_dim))
		self.Wy = np.random.uniform(-bound_2, bound_2, (self.word_dim, self.hidden_dim))

		self.bh = np.zeros((self.hidden_dim,1))
		self.by = np.zeros((self.word_dim,1))
	def forward_propagation(self,x):
		# input = x is a single sentence
		# output = y_hat is a single sentence
		n = len(x)
		h = np.zeros((n+1, self.hidden_dim))
		h[-1] = np.zeros(self.hidden_dim)
		y_hat = np.zeros((n,self.word_dim))
		for t in np.arange(n):
			h[t] = np.tanh(self.Wx[:,x[t]] + np.dot(self.Wh, h[t-1]))
			y_hat[t] = softmax(np.dot(self.Wy, h[t]))
		return h,y_hat
	def total_loss_op (self, x, y):
		# input = entire corpus: X_train, y_train
		# output = total loss over whole training examples
		loss = 0
		n = len(y)
		for t in np.arange(n):
			h,y_hat = self.forward_propagation(x[t])
			preds = y_hat[np.arange(len(y[t])), y[t]]
			loss += -1 * np.sum(np.log(preds))
		return loss
	def loss_op (self, x, y):
		# input = entire training example X_train, y_train
		# output: total loss divided by number of words
		N = np.sum([len(i) for i in y])
		return self.total_loss_op(x,y)/N
	def back_propagation(self, x, y):
		# T = len(y)
	 #    # Perform forward propagation
		# s, o = self.forward_propagation(x)
	 #    # We accumulate the gradients in these variables
		# dLdU = np.zeros(self.Wx.shape)
		# dLdV = np.zeros(self.Wy.shape)
		# dLdW = np.zeros(self.Wh.shape)
		# delta_o = o
		# delta_o[np.arange(len(y)), y] -= 1.
	 #    # For each output backwards...
		# for t in np.arange(T)[::-1]:
		# 	dLdV += np.outer(delta_o[t], s[t].T)
	 #        # Initial delta calculation
		# 	delta_t = self.Wy.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
	 #        # Backpropagation through time (for at most self.bptt_truncate steps)
		# 	for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
	 #            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
		# 		dLdW += np.outer(delta_t, s[bptt_step-1])              
		# 		dLdU[:,x[bptt_step]] += delta_t
	 #            # Update delta for next step
		# 		delta_t = self.Wh.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
		# return [dLdU, dLdW, dLdV]

		# input = a single sentence x, a single sentence y
		# output = dWx, dWh, dWy
		n = len(y)
		h, y_hat = self.forward_propagation(x)
		
		dWx = np.zeros_like(self.Wx)
		dWh = np.zeros_like(self.Wh)
		dWy = np.zeros_like(self.Wy)

		# delta_y = y_hat - y
		delta_y = y_hat
		delta_y[np.arange(len(y)), y] -= 1

		for t in np.arange(n)[::-1]:
			dWy += np.outer(delta_y[t], h[t].T)
			# Loss function derivative wrt h[t]
			# dL/dH = (Wy * delta_y[t])
			dL_dH = self.Wy.T.dot(delta_y[t])
			# Ht derivative wrt z[t] where z[t] = Wx * x[t] + Wh * h[t-1]
			# dH/dZ = 1 - h[t] ** 2
			dH_dZ = (1 - (h[t] ** 2))
			# dL/dZ
			dL_dZ = dL_dH * dH_dZ
			for bptt_step in np.arange (max(0, t - self.bptt_truncate), t+1)[::-1]:
				# note: dZ/dWh = h[t-1]
				# dL/dWh = dL/dZ * dZ/dWh
				dWh += np.outer(dL_dZ, h[bptt_step-1])
				# note: dZ/dWx = x[t]
				# dL/dWx = dL/dZ * dZ/dWx
				dWx [:,x[bptt_step]] += dL_dZ

				# update derivative for the next step t-1
				dL_dH = self.Wh.T.dot(dL_dZ)
				dH_dZ = 1 - h[bptt_step-1] ** 2
				dL_dZ = dL_dH * dH_dZ
		return [dWx, dWh, dWy]
	def gradient_checking (self, x, y, delta = 0.001, threshold = 0.01):
		# input: x and y are single sentence
		# calculate gradients by back_progapagation
		gradients = self.back_propagation(x,y)
		# Wx, Wh, Wy parameters
		model_params = ['Wx', 'Wh', 'Wy']
		# gradient checking for each paramter
		for pidx, pname in enumerate(model_params):
			param = operator.attrgetter(pname)(self)
			# Iterate over each element of param matrix
			it = np.nditer(param, flags = ['multi_index'], op_flags = ['readwrite'])
			while not it.finished:
				
				idx = it.multi_index
				# get original value of param before modifying
				original_value = param[idx]

				param[idx] = original_value - delta
				left_value = self.total_loss_op([x], [y])
				param[idx] = original_value + delta
				right_value = self.total_loss_op([x], [y])

				approx_gradients = (right_value - left_value) / (2*delta)
				backprop_gradients = gradients[pidx][idx]

				relative_error = np.abs(approx_gradients - backprop_gradients)
				# if relative error is too large, greater than threshold, then gradient checking fails
				if relative_error > threshold:
					print ('Gradient Checking FAILED')
					return
				param[idx] = original_value
				it.iternext()
			print('Gradient Checking of %s is passed' %(pname))
	def sgd(self, x, y, learning_rate):
		# input: x and y are single sentence
		dWx, dWh, dWy = self.back_propagation(x,y)
		self.Wx -= learning_rate * dWx
		self.Wh -= learning_rate * dWh
		self.Wy -= learning_rate * dWy
	def train_op (self, X_train, y_train, nepoch = 100):
		# input: entire training examples
		# process: training each example with sgd through a number of epochs
		# output: none
		losses = []
		n_examples_seen = 0
		for epoch in np.arange(nepoch):
			if (epoch%10 == 0):
				loss = self.loss_op(X_train, y_train)
				losses.append((loss, n_examples_seen))
				print ('At epoch %d, loss calculated is %f' %(epoch,loss))
				if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
					self.learning_rate *= 0.5
					print ('Set learning rate to %.4f' %self.learning_rate)
			sys.stdout.flush()
			# train with sgd for each example
			for i in np.arange(len(y_train)):
				self.sgd(X_train[i], y_train[i], self.learning_rate)
				n_examples_seen += 1
	def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
	    # We keep track of the losses so we can plot them later
	    losses = []
	    num_examples_seen = 0
	    for epoch in range(nepoch):
	        # Optionally evaluate the loss
	        if (epoch % evaluate_loss_after == 0):
	            loss = self.calculate_loss(X_train, y_train)
	            losses.append((num_examples_seen, loss))
	            # _time = time.time().strftime('%Y-%m-%d %H:%M:%S')
	            # print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (_time, num_examples_seen, epoch, loss))
	            # Adjust the learning rate if loss increases
	            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
	                learning_rate = learning_rate * 0.5 
	                print ("Setting learning rate to %f" % learning_rate)
	            sys.stdout.flush()
	        # For each training example...
	        for i in range(len(y_train)):
	            # One SGD step
	            self.sgd_step(X_train[i], y_train[i], learning_rate)
	            num_examples_seen += 1
if __name__ == '__main__':
	model = RNN_numpy(100,0.1,4)
	_, y_hat = model.forward_propagation(X_train[9])
	print(model.loss_op(X_train,y_train))
	model.gradient_checking(X_train[10], y_train[10])
	model.train_op(X_train,y_train)






