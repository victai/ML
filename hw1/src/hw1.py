import numpy as np
import scipy
import pandas as pd
import sys
np.set_printoptions(threshold=np.inf)

def rmse(prediction , actual):
	return np.sqrt(((prediction - actual)**2).mean())

def read_train():
	df = pd.read_csv('data/train.csv', encoding='Big5').replace('NR', '0').convert_objects(convert_numeric=True)
	x_data = df.iloc[:, 3:].values.reshape((240, -1))
	#x_data[:, 24*10:24*11] = x_data[:, :24] * x_data[:, 24*9:24*10] #pm2.5*temp
	x2 = x_data ** 2
	x_data = np.hstack((x_data, x2))
	B = x_data[0].reshape(36, -1)
	
	for i in range(1, 240):
		A = x_data[i].reshape(36, -1)
		B = np.concatenate((B, A), axis=1)
	Y = B.T[:, 9]
	for i in range(11, -1, -1):
		for j in range(9):
			Y = np.delete(Y, 480*i)									#Y.shape = (5652, 1)
	#for i in range(36):
	#	B[i] = (B[i] - np.mean(B[i]))/np.std(B[i], dtype=np.float64)
	X = B.T[0:9].reshape(-1, )
	X = [X]
	for i in range(12):
		for j in range(471):
			if i == 0 and j == 0:
				continue
			A = B.T[480*i+j : 480*i+j+9].reshape(-1, )
			X = np.concatenate((X, [A]), axis=0)					#X.shape = (5652, 324)
	return (X, Y)
	

def read_test():
	df_test = pd.read_csv(sys.argv[1], encoding="Big5", header=None).replace('NR', '0').convert_objects(convert_numeric=True)
	test_x = df_test.iloc[:, 2:].values.reshape((240, -1))
	#test_x[:, 9*10:9*11] = test_x[:, :9] * test_x[:, 9*9:9*10]
	x2 = test_x**2
	test_x = np.hstack((test_x, x2))
	B2 = test_x[0].reshape(36, -1)
	for i in range(1, 240):
		A2 = test_x[i].reshape(36, -1)
		B2 = np.concatenate((B2, A2), axis=0)
	#B2 = (B2-np.mean(B2))/np.std(B2)
	X = B2[:36, :].T.reshape(-1, )

	for i in range(1, 240):
		A2 = B2[36*i:36*i+36, :].T.reshape(-1, )
		X = np.vstack((X, A2))
	return X

def Test(X, w, b):
	test_pred = X.dot(w)
	index_name = ['id_' + str(i) for i in range(240)]
	final_df = pd.DataFrame(index_name, columns=['id'])
	final_df['value'] = pd.DataFrame(test_pred)
	final_df.to_csv(sys.argv[2], index=False)

def Regression(X, Y):
	w = np.array([[0.0] for i in range(324)])
	Delta = np.zeros([324, 1])
	for i in range(18):
		Delta[18*i+9, 0] = 1
		Delta[18*i, 0] = 1
		Delta[18*i+8, 0] = 1
		#Delta[18*i+10, 0] = 1
	
	lr = 0.001
	iteration = 200000
	Lamda = 0 
	b = 0
	Y = Y.reshape((Y.shape[0], 1))
	sum_w_grad = np.zeros((324, 1))
	sum_b_grad = 0
	for i in range(471*6, 471*7):
		Y[i] = 0
		X[i] = np.zeros([1, 324])
	
	for it in range(iteration):
		f = X.dot(w*Delta)
		w_grad = -2*(X.T.dot(Y - f) + Lamda * w**2)
		b_grad = -2*np.sum(Y-f)
		sum_w_grad += np.square(w_grad)
		sum_b_grad += np.square(b_grad)

		for i in range(9):
			w -= lr*w_grad/np.sqrt(sum_w_grad)*Delta
			b -= lr*b_grad/np.sqrt(sum_b_grad)*Delta

		loss = 0
		pred = X.dot(w)
		loss = rmse(pred, Y)
		if it % 100 == 0: print('\r', loss, end='')

	print()

	return (w, b)

if __name__ == '__main__':
	#X, Y = read_train()
	X_test = read_test()
	#w, b = Regression(X, Y)
	#w = np.append(w, b, axis=1)
	#model_df = pd.DataFrame(w)
	#model_df.to_csv('result/parameters.csv', index=False)
	w = pd.read_csv('result/parameters_best.csv').iloc[:, 0]
	b = pd.read_csv('result/parameters_best.csv').iloc[:, 1]
	Test(X_test, w, b)
