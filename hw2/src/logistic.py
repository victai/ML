import numpy as np
import pandas as pd
import sys
import time
np.set_printoptions(threshold=np.inf)

dim = 141

def sigmoid(z):
	return np.clip(1/(1+ np.exp(-z)), 1e-8, 1-(1e-8))

def Read_data():
	X_all = pd.read_csv(sys.argv[1], encoding='Big5').values
	Y_all = pd.read_csv(sys.argv[2], encoding='Big5').values
	X_test = pd.read_csv(sys.argv[3], encoding='Big5').values
	X_all = np.hstack((X_all, X_all[:, :5]**2, X_all[:, :5]**3, X_all[:, :5]**4, X_all[:, :5]**5, \
			X_all[:, :5]**6, X_all[:, :5]**7, X_all[:, :5]**8))
	X_test = np.hstack((X_test, X_test[:, :5]**2, X_test[:, :5]**3, X_test[:, :5]**4, X_test[:, :5]**5, \
			X_test[:, :5]**6, X_test[:, :5]**7, X_test[:, :5]**8))
	X_all_test = np.concatenate((X_all, X_test))
	mu = sum(X_all_test)/ (X_all_test.shape[0])
	sigma = np.std(X_all_test, axis=0)
	X_all = (X_all-mu)/sigma
	X_test = (X_test-mu)/sigma

	return X_all, Y_all, X_test

def Shuffle(X, Y):
	randomize = np.arange(Y.shape[0])
	np.random.shuffle(randomize)
	return X[randomize], Y[randomize]

def Split_validation_set(X_train, Y_train, percentage):
	X_train, Y_train = Shuffle(X_train, Y_train)
	val_data_size = int(np.floor(Y_train.shape[0] * percentage))
	
	return X_train[val_data_size:, :], Y_train[val_data_size:, :], X_train[0:val_data_size, :], Y_train[0:val_data_size, :]

def Validate(X_val, Y_val, w, b):
	f = sigmoid(X_val.dot(w)+b)
	pred = np.zeros([X_val.shape[0], 1], dtype=int)
	pred[f > 0.5] = 1
	print('Validation Accuracy: ', np.count_nonzero(pred==Y_val)/Y_val.shape[0])

def Test(X_test, w, b):
	f = sigmoid(X_test.dot(w)+b)
	pred = np.zeros([X_test.shape[0], 1], dtype=int)
	pred[f > 0.5] = 1
	index_name = [i for i in range(1,X_test.shape[0]+1)]
	df = pd.DataFrame(index_name, columns=['id'])
	df['label'] = pd.DataFrame(pred)
	df.to_csv(sys.argv[4], index=False)

def Logistic_regression(X_train, Y_train):
	w = np.zeros([dim, 1])
	b = 0
	sum_w_grad = np.zeros([dim, 1])
	sum_b_grad = 0
	Lamda = 0
	iteration = 2000
	lr = 0.1
	for i in range(iteration):
		f = sigmoid(X_train.dot(w) + b)
		w_grad = -X_train.T.dot(Y_train-f) - Lamda*w
		b_grad = -np.sum(Y_train-f)
		sum_w_grad += np.square(w_grad)
		sum_b_grad += np.square(b_grad)
		w -= lr*w_grad/np.sqrt(sum_w_grad)
		b -= b_grad/np.sqrt(sum_b_grad)
		pred = np.zeros([X_train.shape[0], 1], dtype=int)
		pred[f > 0.5] = 1	
		if i % 10 == 0:	print('\r', i, np.count_nonzero(pred==Y_train)/Y_train.shape[0], end='')
		X_train, Y_train = Shuffle(X_train, Y_train)
	print()
	return w, b

if __name__ == '__main__':
	X_all, Y_all, X_test= Read_data()
	#X_train, Y_train, X_val, Y_val = Split_validation_set(X_all, Y_all, 0.1)
	w, b = Logistic_regression(X_all,Y_all)
	#Validate(X_val, Y_val, w, b)
	idx = np.arange(dim).reshape(dim,1)
	idx = np.concatenate((idx, w), axis=1)
	#print(idx)
	Test(X_test, w, b)
