import numpy as np
import tensorflow as tf
import pickle
import argparse
from collections import Counter
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, Embedding, Flatten
from keras.optimizers import Adam

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

BATCH_SIZE=100
vocab=125659  #train: 82946

def RNN(X_train, Y_train):
	model = Sequential()
	model.add(Dense(input_dim=7643, units=100, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(units=64, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(units=64, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(units=1, activation='sigmoid'))
	
	model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=10, validation_split=0.1, shuffle=True)
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	with open('../history/bow_history.pkl', 'wb') as f:
		pickle.dump((acc, val_acc), f)
	model.save('bow_model.h5')
	score = model.evaluate(X_train, Y_train)
	print('Total loss: ', score[0])
	print('Accuracy: ', score[1])

def read_data():
	with open('../data/training_label.txt', 'r') as f:
		labeled_data = f.read().split('\n')[:-1]
	with open('common_over30.txt', 'r') as f:
		Most_Common = f.read().split()
	X_train = [None]*200000
	Y_train = np.zeros(200000, dtype=int)
	for i in range(200000):
		X_train[i] = labeled_data[i][10:].split()
		Y_train[i] = int(labeled_data[i][0])
	count = np.zeros((200000, len(Most_Common)))
	for i in range(200000):
		for j in X_train[i]:
			if j in Most_Common:
				count[i][Most_Common.index(j)] += 1

	return count, Y_train

def get_most_frequent():
	with open('../data/training_label.txt', 'r') as f:
		labeled_data = f.read().split('\n')[:-1]
	with open('../data/testing_data.txt', 'r') as f:
		test_data = f.read().split('\n')[:-1]
	for i in range(200000):
		labeled_data[i] = labeled_data[i][10:].split()
		test_data[i] = test_data[i].split(',', 1)[1].split()
	all_data = labeled_data + test_data

	flat_data = [j for i in all_data for j in i]
	Most_Common = Counter(flat_data).most_common(20000)
	keys = []
	for key, value in Most_Common:
		if(value > 30):	keys.append(key)
	with open('common_over30.txt', 'w') as f:
		for key in keys:
			f.write(key+" ")

def main():
	get_most_frequent()
	X_train, Y_train = read_data()
	RNN(X_train, Y_train)


if __name__ == '__main__':
	main()
