import platform
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from itertools import chain
import time
import numpy as np
vocab_size = 8000
unknown_token = "UNKNOWN_TOKEN"
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"
start = time.time()
print ("Reading data ...")
filename = 'LoR_full' 
if platform.system() == 'Linux':
	filepath = os.getcwd() + '/Data/'
else:
	filepath= os.getcwd() + '\\Data\\'
# Step 1: read data from files
# using nltk tokenize to read data
text = nltk.corpus.gutenberg.raw(filepath+filename)
sents = nltk.sent_tokenize(text)
sentences = ["%s %s %s" %(start_token, x, end_token) for x in sents]
words = nltk.word_tokenize(text)
word_freq = nltk.FreqDist(words)
vocab = word_freq.most_common(vocab_size-1)
# tokenized_sentences = nltk.word_tokenize(sents)
print('Corpus has %d sentences. The second sentence is: \n %s' %(len(sentences),sentences[1]))
print('It took {:.2f} seconds to read data'.format(time.time() - start))
print ('It found {:} unique word tokens.'.format(len(word_freq.items())))
print ('Ten most freq words are \n ', vocab[:10] )

# using python file open to read data
# data = open(filepath + filename, 'r', encoding='utf-8')
# corpus = list(data)
# #tokenize text data into words, this is list of list
# tokenized_sentences = [nltk.word_tokenize(line) for line in corpus] 
# #words = list(chain.from_iterable(tokenized_sentences)) #convert list of list to a single list of words
# word_freq = nltk.FreqDist(chain(*tokenized_sentences))
# vocab = word_freq.most_common(vocab_size-1)
# print('It took {:.2f} seconds to read data'.format(time.time() - start))
# print ('It found {:} unique word tokens.'.format(len(word_freq.items())))
# print ('Ten most freq words are \n ', vocab[:10] )

# using nltk PlaintextCorpusReader
# corpus0 = PlaintextCorpusReader(filepath, "MyText.txt")
# corpus = nltk.Text(corpus0.words())
# print(corpus[:10])

# Step 2: create dictionary to access data and remove not frequent words
#create dictionary word to index and index to word

idx_to_word = [word[0] for idx,word in enumerate(vocab)] # create a list for efficiency
idx_to_word.append(unknown_token)
word_to_idx = {word:idx for idx,word in enumerate(idx_to_word)}
word_to_idx = dict([(w,i) for i,w in enumerate(idx_to_word)])
print('The 10th most frequent word is: %s' %(idx_to_word[9]))
print('The word: {:} is the {:}th most frequent word'.format('was', word_to_idx['was'] + 1))

#X_train = np.asarray([[word_to_idx[w] for w in sentence[:-1]] for sentence in sentences])

start = time.time()
# replace word not in our vocabulary with unknown_token
for i,sentence in enumerate(sentences):
	sentences[i] = [w if w in word_to_idx else unknown_token for w in sentence]
print("Finished replace not frequent words with unknown_token took {:.2f} seconds".format(time.time() - start))

# Step 3: create training data
X_train = np.asarray([[word_to_idx[w] for w in sentence[:-1]] for sentence in sentences])
y_train = np.asarray([[word_to_idx[w] for w in sentence[1:]] for sentence in sentences])

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x), axis=0)
class RNNNumpy:
	def __init__(self,word_dim,hidden_dim,bptt_truncate):
		self.hidden_dim = hidden_dim
		self.word_dim = word_dim
		self.bptt_truncate = bptt_truncate
		self.Wx, self.Wh. self.Wy = self.create_weights()
	def createWeights(self):
		bound_1 = np.sqrt(1/self.word_dim)
		bound_2 = np.sqrt(1/self.hidden_dim)
		Wx = np.random.uniform(-bound_1, bound_1, (self.hidden_dim, self.word_dim))
		Wh = np.random.uniform(-bound_2, bound_2, (self.hidden_dim, self.hidden_dim))
		Wy = np.random.uniform(-bound_2, bound_2, (self.word_dim, self.hidden_dim))
		bh = np.zeros(self.hidden_dim,1)
		by = np.zeros(self.word_dim,1)
		return Wx, Wh, Wy
	def forward_propagation(self,x):
		T = len(x)
		h = np.zeros((T+1, self.hidden_dim))
		h[-1] = np.zeros(self.hidden_dim)
		y_hat = np.zeros((T,self.word_wim))
		for t in xrange(T):
			h[t] = np.tanh(np.dot(self.Wx,x) + np.dot(self.Wh,h[t-1]) + self.bh)
			y_hat[t] = softmax(np.dot(self.Wy,h[t]) + self.by)
		return h,y_hat
	def predict(self,x):
		h,y_hat = self.feedforward_propagation(x)
		return np.argmax(y_hat,axis=1) #get maximum index of each row
if __name__ == "main":
	model = RNNNumpy(300,100,4)