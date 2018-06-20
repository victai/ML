import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv
#np.set_printoptions(threshold=np.inf)

dim = 111

def Sigmoid(z):
	z = 1/(1+ np.exp(-z))
	return np.clip(z, 1e-8, 1-(1e-8))

def Normalize(X_train, X_test):
	X_train_test = np.concatenate((X_train, X_test))
	mu = sum(X_train_test)/ (X_train_test.shape[0])
	sigma = np.std(X_train_test, axis=0)
	return (X_train-mu)/sigma, (X_test-mu)/sigma	

def Read_data():
	X_train = pd.read_csv(sys.argv[1], header=0).values
	Y_train = pd.read_csv(sys.argv[2], header=0).values
	X_test = pd.read_csv(sys.argv[3], header=0).values
	X_train = np.hstack((X_train, X_train[:, :5]**2))
	X_test = np.hstack((X_test, X_test[:, :5]**2))
	return X_train, Y_train, X_test

def Test(X_test, w,b):
	f = Sigmoid(X_test.dot(w)+b)
	result = np.zeros([X_test.shape[0], 1], dtype=int)
	result[f > 0.5] = 1
	index_name = [i for i in range(1,X_test.shape[0]+1)]
	df = pd.DataFrame(index_name, columns=['id'])
	df['label'] = pd.DataFrame(result)
	df.to_csv(sys.argv[4], index=False)

def Generative(X_train, Y_train):
	mu1 = np.zeros([dim,1])
	mu2 = np.zeros([dim,1])
	X_1 = X_train[Y_train[:,0]==1, :]
	X_2 = X_train[Y_train[:,0]==0, :]
	cnt1 = X_1.shape[0]
	cnt2 = X_2.shape[0]
	mu1 = sum(X_1) / cnt1
	mu2 = sum(X_2) / cnt2
	sigma1 = np.zeros([dim,dim])
	sigma2 = np.zeros([dim,dim])
	for i in range(X_1.shape[0]):
		sigma1 += np.dot(np.transpose([X_1[i] - mu1]), [X_1[i]-mu1])
	for i in range(X_2.shape[0]):
		sigma2 += np.dot(np.transpose([X_2[i] - mu2]), [X_2[i]-mu1])
	sigma = (sigma1*cnt1 + sigma2*cnt2) / (cnt1 + cnt2)
	sigma_inverse = inv(sigma)
	w = (mu1-mu2).dot(sigma_inverse)
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(cnt1)/cnt2)
	f = Sigmoid(X_train.dot(w) + b)
	f = np.around(f).reshape(-1,1)
	result = (f == Y_train)
	result = np.squeeze(result)
	print('Train Accuracy: ', sum(result)/result.shape[0])

	return w, b

if __name__ == '__main__':
	X_train, Y_train, X_test = Read_data()
	X_train, X_test = Normalize(X_train, X_test)
	w, b = Generative(X_train, Y_train)
	Test(X_test, w,b)
