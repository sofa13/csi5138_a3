import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Dense, Flatten
from keras.layers import Input, LSTM, Dropout, SimpleRNN
from keras.models import Sequential, Model
from keras import optimizers

import matplotlib
import matplotlib.pyplot as plt

import os
import numpy as np

from pprint import pprint

# read from file
with open('gensim', 'r', encoding="utf-8") as fin:
    content = eval(fin.read())

Xtrain, ytrain, Xtest, ytest = content

print(Xtrain[0][:5])
print(ytrain[0])
print(Xtest[0][:5])
print(ytest[0])
print(Xtrain[12500][:5])
print(ytrain[12500])
print(Xtest[12500][:5])
print(ytest[12500])
print("# Xtrain: ", len(Xtrain))
print("# ytrain: ", len(ytrain))
print("# Xtest: ", len(Xtest))
print("# ytest: ", len(ytest))

X = list(Xtrain + Xtest)
y = list(ytrain + ytest)
print("# X: ", len(X))
print("# y: ", len(y))

# convert reviews to pos and negative
# pos 1, neg 0
print(y[:10])
print(y[-10:])
y = [int(a)>= 7 for a in y]
print(y[:10])
print(y[-10:])

MAX_SEQUENCE_LENGTH=500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels[0])

embeddings_index = {}
glove_file = './glove.6B/glove.6B.50d.txt'

with open(glove_file, "r", encoding='utf-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=50
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

print(len(word_index))
print(len(embedding_matrix))

#print(list(word_index.items())[100])
# sentence word_index = 101
print(word_index['king'])
print(word_index['queen'])
print(embedding_matrix[682][:5])
print(embedding_matrix[1577][:5])
print(embedding_matrix[101][:5])
print(embedding_matrix[0][:5])

# separate train, val and test. Take test as half val, half test.
X_train, y_train = data[:25000], labels[:25000]
X_test, y_test = data[25000:], labels[25000:]

def vanilla_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
	model = Sequential()
	model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False, weights=[embedding_matrix]))
	model.add(SimpleRNN(units=state, input_shape=(num_words,1), return_sequences=False))
	model.add(Dropout(dropout))
	model.add(Dense(num_outputs, activation='sigmoid'))
	
	rmsprop = optimizers.RMSprop(lr = lra)
	model.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics = ['accuracy'])
	
	return model
	
def lstm_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
	model = Sequential()
	model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False, weights=[embedding_matrix]))
	model.add(LSTM(state))
	model.add(Dropout(dropout))
	model.add(Dense(num_outputs, activation='sigmoid'))
	
	rmsprop = optimizers.RMSprop(lr = lra)
	model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
	
	return model
	
def runModel(state, lr, batch, dropout, model, epoch=5, num_outputs=2, emb_dim=100, input_length=2380):
		
	num_words = len(word_index)
	if model == "lstm": 
		model = lstm_rnn(num_words, state, lr, dropout)
	elif model == "vanilla":
		model = vanilla_rnn(num_words, state, lr, dropout)
		
	#model.summary()
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=0)

	testscore = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', testscore[0])
	print('Test accuracy:', testscore[1])
	
	return [history.history, testscore]

def hypruns(state, comb, repeats, model):
	history = []
	testscore = []

	for i in range(repeats):
		l, b, d = comb
		print("state %s, lr %s, batch %s, dropout %s, model %s." %(state, l, b, d, model))
		res = runModel(state, l, b, d, model)
		
		if res:
			history.append(res[0])
			testscore.append(res[1])
	
	# take avg of testscore
	testscore = list(np.mean(testscore, axis=0))
	hyps = [state] + comb
	
	return [history, testscore, hyps]
	
def tunehyps(states, comb, repeats, model):
	for i, comb in enumerate(combs):
		for state in states:
			history, testscore, hyps = hypruns(state, comb, repeats, model)
				
			# save testscore to file
			with open('./experiments/'+model+'/'+str(state)+'/testscore_'+'comb_'+str(i), 'w', encoding="utf-8") as fout:
				pprint(history, fout)

			# save history to file
			with open('./experiments/'+model+'/'+str(state)+'/history_'+'comb_'+str(i), 'w', encoding="utf-8") as fout:
				pprint(testscore + hyps, fout)
	
states = [20, 50, 100, 200, 500]
lrs = [0.1, 0.01, 0.001]
batches = [100, 200, 500]
dropouts = [0.1, 0.2, 0.5]
#epochs = [5, 10, 15]
repeats = 1
models = ["lstm", "vanilla"]

numComb = 9
combs = []
np.random.seed(42)

for i in range(numComb):
	combs.append([np.random.choice(lrs), np.random.choice(batches), np.random.choice(dropouts)])

print(combs)
for model in models:		
	tunehyps(states, combs, repeats, model)



