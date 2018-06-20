import numpy as np
import argparse
import ipdb
import pickle
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Input, merge, Flatten, Dropout
from keras.layers import Add, Reshape, Concatenate, Dense
from keras.layers.normalization import BatchNormalization


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

mu = std = 0
dim = 96
genres = {'Animation':0, 'Children\'s':1, 'Comedy':2, 'Adventure':3, 'Fantasy':4, 'Romance':5, 'Drama':6, 'Action':7, 'Crime':8, 'Thriller':9, 'Horror':10, 'Sci-Fi':11, 'Musical':12, 'Mystery':13, 'Documentary':14, 'War':15, 'Film-Noir':16, 'Western':17}

def read_data(args):
	X_train = np.zeros((899873, 21), dtype=int)
	Y_train = np.zeros(899873, dtype=int)
	X_test = np.zeros((100336, 21), dtype=int)
	user_ids = [None]*6040
	movie_ids = np.zeros((3883, 19), dtype=int)
	
	with open(args.users_data, 'r') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		user_ids[i] = line.split('::')[:-1]
		user_ids[i][1] = 0 if (user_ids[i][1] == 'F') else 1
	user_ids = np.array(user_ids)

	with open(args.movies_data, 'r', encoding='utf-8', errors='ignore') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		movie_ids[i][0] = line.split('::')[0]
		cat = line.split('::')[2].split('|')
		for c in cat:	movie_ids[i][genres[c]+1] = 1

	with open(args.training_data, 'r') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		u = line.split(',')[1]
		m = int(line.split(',')[2])
		X_train[i][:3] = user_ids[np.where(user_ids[:,0] == u)[0][0]][1:]
		X_train[i][3:] = movie_ids[np.where(movie_ids[:,0] == m)[0][0]][1:]
		Y_train[i] = line.split(',')[3]

	with open(args.testing_data, 'r') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		u = line.split(',')[1]
		m = int(line.split(',')[2])
		X_test[i][:3] = user_ids[np.where(user_ids[:,0] == u)[0][0]][1:]
		X_test[i][3:] = movie_ids[np.where(movie_ids[:,0] == m)[0][0]][1:]
	print(X_test[:5])

	return X_train, Y_train, X_test

def main(args):
	X_train, Y_train, X_test = read_data(args)

	idx = np.arange(X_train.shape[0])
	np.random.shuffle(idx)
	X_train = X_train[idx]
	Y_train = Y_train[idx]

	model = Sequential()
	model.add(Dense(units=64, input_dim=21, activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(units=64, activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.3))
	model.add(Dense(units=64, activation='relu'))
	#model.add(BatchNormalization())
	model.add(Dense(units=64, activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.3))
	model.add(Dense(units=1, activation='relu'))

	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.1, shuffle=True)

	model.save(args.model_path)
		
	model = load_model(args.model_path)
	model.summary()
	Y = model.predict(X_test)
	with open(args.output, 'w') as f:
		f.write('TestDataID,Rating\n')
		for i, pred in enumerate(Y):
			f.write(str(i+1)+','+str(pred[0])+'\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_data', type=str)
	parser.add_argument('--testing_data', type=str)
	parser.add_argument('--users_data', type=str)
	parser.add_argument('--movies_data', type=str)
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--history', type=str)
	parser.add_argument('--output', type=str)
	args = parser.parse_args()
	main(args)
