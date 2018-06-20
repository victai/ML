import numpy as np
import argparse
import ipdb
import pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Input, merge, Flatten, Dropout
from keras.layers import Add, Reshape

mu = std = 0
dim = 96

def read_data(args):
	X_train = np.zeros((899873, 3), dtype=int)
	Y_train = np.zeros(899873, dtype=int)
	X_test = np.zeros((100336, 3), dtype=int)
	user_ids = np.zeros(6040, dtype=int)
	movie_ids = np.zeros(3883, dtype=int)
	#with open(args.training_data, 'r') as f:
	#	data = f.read().split('\n')[1:-1]
	#for i, line in enumerate(data):
	#	X_train[i] = line.split(',')[:3]
	#	Y_train[i] = line.split(',')[3]
	#global mu, std
	#mu = np.mean(Y_train)
	#std = np.std(Y_train)
	#Y_train = (Y_train - mu) / std

	with open(args.testing_data, 'r') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		X_test[i] = line.split(',')

	with open(args.users_data, 'r') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		user_ids[i] = line.split('::')[0]
	
	with open(args.movies_data, 'r', encoding='utf-8', errors='ignore') as f:
		data = f.read().split('\n')[1:-1]
	for i, line in enumerate(data):
		movie_ids[i] = line.split('::')[0]

	return X_train, Y_train, X_test, user_ids, movie_ids

def main(args):
	X_train, Y_train, X_test, user_ids, movie_ids = read_data(args)
	user_train = X_train[:,1]
	movie_train = X_train[:,2]
	user_test = X_test[:,1]
	movie_test = X_test[:,2]
	
	user_train = np.array([np.where(user_ids == i) for i in X_train[:, 1]]).flatten()
	movie_train = np.array([np.where(movie_ids == i) for i in X_train[:, 2]]).flatten()
	user_test = np.array([np.where(user_ids == i) for i in X_test[:, 1]]).flatten()
	movie_test = np.array([np.where(movie_ids == i) for i in X_test[:, 2]]).flatten()
	X_test = [user_test, movie_test]
	'''
	idx = np.arange(user_train.shape[0])
	np.random.shuffle(idx)
	user_train = user_train[idx]
	movie_train = movie_train[idx]
	Y_train = Y_train[idx]

	X_train = [user_train, movie_train]
	X_val = [user_train[-90000:], movie_train[-90000:]]
	Y_val = Y_train[-90000:]
	X_train = [user_train[:-90000], movie_train[:-90000]]
	Y_train = Y_train[:-90000]

	bias = np.zeros(user_train.shape[0])

	user_in = Input(shape=(1,))
	user_vec = Flatten()(Embedding(6041, dim)(user_in))
	user_vec = Dropout(0.333)(user_vec)

	movie_in = Input(shape=(1,))
	movie_vec = Flatten()(Embedding(3884, dim)(movie_in))
	movie_vec = Dropout(0.3)(movie_vec)

	dot = merge([user_vec, movie_vec], mode='dot')
	u_bias = Reshape((1,))(Embedding(6041,1)(user_in))
	m_bias = Reshape((1,))(Embedding(3884,1)(movie_in))
	add_bias = Add()([dot, u_bias, m_bias])

	model = Model([user_in, movie_in], add_bias)
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X_train, Y_train, epochs=75, batch_size=10000, validation_data=(X_val, Y_val))
	'''
	'''	
	with open(args.history, 'wb') as f:
		pickle.dump((history.history['loss'], history.history['val_loss']), f)
	'''
	#model.save(args.model_path)
		
	model = load_model(args.model_path)
	model.summary()
	Y = model.predict(X_test)
	with open(args.output, 'w') as f:
		f.write('TestDataID,Rating\n')
		for i, pred in enumerate(Y):
			#ans = pred[0]*std + mu
			#f.write(str(i+1)+','+str(ans)+'\n')
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
