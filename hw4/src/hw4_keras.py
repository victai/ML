import numpy as np
import tensorflow as tf
import pickle
import argparse
from gensim.models import Word2Vec
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, Embedding, Flatten
from keras.optimizers import Adam
from random import shuffle

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

BATCH_SIZE=256
EPOCHS=50

def RNN(X_train, Y_train, args, weights):
	model = Sequential()
	model.add(Embedding(input_dim=55777, output_dim=100, input_length=39, weights=[weights], trainable=False))
	model.add(LSTM(
		#input_shape=(39, 100),
		output_dim=64,
		dropout=0.4,
		recurrent_dropout=0.4,
		return_sequences=True
	))
	model.add(LSTM(
		output_dim=64,
		dropout=0.4,
		recurrent_dropout=0.4,
		return_sequences=True
	))
	model.add(LSTM(
		output_dim=64,
		dropout=0.4,
		recurrent_dropout=0.4
	))
	model.add(Dense(units=64, activation='relu'))
	model.add(Dense(units=1, activation='sigmoid'))
	
	model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, shuffle=True)
	
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	with open('history/new_history.pkl', 'wb') as f:
		pickle.dump((acc, val_acc), f)
	
	
	model.save(args.model_save_path)
	score = model.evaluate(X_train, Y_train)
	print('Total loss: ', score[0])
	print('Accuracy: ', score[1])
	
def read_data(args):
	with open(args.labeled_path, 'r') as f:
		data = f.read().split('\n')[:-1]
	X_train = [None]*200000
	Y_train = np.zeros(200000, dtype=int)
	for i, line in enumerate(data):
		X_train[i] = line[10:].split()
		Y_train[i] = int(line[0])
	
	vec_model = Word2Vec.load(args.word2vec_model)
	'''
	weights = np.zeros((55777, 100))
	dictionary = {}
	for i, vocab in enumerate(vec_model.wv.vocab):
		weights[i+1] = vec_model.wv[vocab]
		dictionary[vocab] = i+1
	'''
	with open('weights.pkl', 'rb') as f:
		weights = pickle.load(f)
	with open('dictionary.pkl', 'rb') as f:
		dictionary = pickle.load(f)

	for i in range(200000):
		for j in range(39):
			l = len(X_train[i])
			if j >= l:
				X_train[i].append(0)
			elif X_train[i][j] in vec_model.wv.vocab:
				X_train[i][j] = dictionary[X_train[i][j]]
			else:
				X_train[i][j] = 0 
	'''	
	with open(args.unlabeled_path, 'r') as f:
		unlabeled_data = f.read().split('\n')[:-1]
	new_unlabeled = []
	for i in range(len(unlabeled_data)):
		if len(unlabeled_data[i].split()) <= 39:
			new_unlabeled.append(unlabeled_data[i].split())

	for i in range(len(new_unlabeled)):
		for j in range(39):
			l = len(new_unlabeled[i])
			if j >= l:
				new_unlabeled[i].append(0)
			elif new_unlabeled[i][j] in vec_model.wv.vocab:
				new_unlabeled[i][j] = dictionary[new_unlabeled[i][j]]
			else:
				new_unlabeled[i][j] = 0
	
	model = load_model('models/pre_semi_supervised_model')
	Y_unlabeled = model.predict(new_unlabeled)
	cnt1 = cnt0 = 0
	for i in range(len(new_unlabeled)):
		if Y_unlabeled[i] > 0.9:
			X_train += [new_unlabeled[i]]
			Y_train = np.append(Y_train, 1)
			cnt1 += 1
		elif Y_unlabeled[i] < 0.1:
			X_train += [new_unlabeled[i]]
			Y_train = np.append(Y_train, 0)
			cnt0 += 1

	idx = np.arange(len(X_train))
	shuffle(idx)
	X_train2 = [None]*len(X_train)
	for i in range(len(X_train)):
		X_train2[i] = X_train[idx[i]]
	Y_train = Y_train[idx]
	print(cnt0, cnt1)
	print(len(X_train2), len(Y_train))
	'''
	return X_train, Y_train, weights

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--labeled_path', type=str)
	parser.add_argument('--unlabeled_path', type=str)
	parser.add_argument('--model_save_path', type=str)
	parser.add_argument('--word2vec_model', type=str)
	args = parser.parse_args()
	X_train, Y_train, weights = read_data(args)
	RNN(X_train, Y_train, args, weights)


if __name__ == '__main__':
	main()
