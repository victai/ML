import numpy as np
import scipy
import pandas as pd
np.set_printoptions(threshold=np.inf)

df = pd.read_csv('../data/train.csv', encoding='Big5')
x_data = df.iloc[:, 3:].values.reshape((240, -1))
x_data[x_data == 'NR'] = '0'
b = x_data[0].reshape(18, -1)

for i in range(1, 240):
	a = x_data[i].reshape(18, -1)
	b = np.concatenate((b, a), axis=1)

Y = b.T[:, 9]
for i in range(11, -1, -1):
	for j in range(9):
		Y = np.delete(Y, 480*i)
Y = Y.astype(float)
'''
Y.shape = (5652, 1)
'''
X = b.T[0:9].reshape(-1, )
X = [X]

for i in range(12):
	for j in range(471):
		if i == 0 and j == 0:
			continue
		a = b.T[480*i+j : 480*i+j+9].reshape(-1, )
		X = np.concatenate((X, [a]), axis=0)
X = X.astype(float)
'''
X.shape = (5652, 162)
'''
w = np.zeros([162, 1])
delta = np.zeros([162, 1])
for i in range(9):
	delta[9+18*i, 0] = 1
lr = .05
iteration = 10000
b = 0

def rmse(prediction , actual):
	return np.sqrt(((prediction - actual)**2).mean())


Y = Y.reshape((Y.shape[0], 1))
sum_w_grad = np.zeros((162, 1))
sum_b_grad = 0

for it in range(iteration):
	#b_grad = np.zeros([162, 1])
	#w_grad = np.zeros([162, 1])

	f = X.dot(w*delta)

	w_grad = -2*X.T.dot(Y - f)
	b_grad = -2*np.sum(Y-f)
	
	sum_w_grad += np.square(w_grad)
	sum_b_grad += np.square(b_grad)
	
	#print(w_grad.shape)
	#print((Y-f).shape)
	#print('\r', w_grad , end='')
		#b = b - lr*b_grad
	for i in range(9):
		w -= lr*w_grad/np.sqrt(sum_w_grad)*delta
		b -= lr*b_grad/np.sqrt(sum_b_grad)*delta
	#print(X.dot(b).shape)
	#print('\r', w*delta, end='')

	loss = 0
	pred = X.dot(w)
	loss = rmse(pred, Y)
	#print(pred)
	if it % 100 == 0: print('\r', loss, end='')

print()

def test():
	df_test = pd.read_csv('../data/test.csv', encoding="Big5", header=None)
	#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
	#	    print(df_test)
	test_x = df_test.iloc[:, 2:].values.reshape((240, -1))
	test_x[test_x == 'NR'] = '0'
	b2 = test_x[0].reshape(18, -1)
	for i in range(1, 240):
		a2 = test_x[i].reshape(18, -1)
		b2 = np.concatenate((b2, a2), axis=0)
	T_X = b2[:18, :].T.reshape(-1, )
	for i in range(1, 240):
		a2 = b2[18*i:18*i+18, :].T.reshape(-1, )
		T_X = np.vstack((T_X, a2))
	T_X = T_X.astype(float)
	test_pred = T_X.dot(w)
	#test_pred = test_pred.tolist()
	#for i in range(240):
	#	test_pred[i] = ('id_'+str(i)+','+str(test_pred[i]).strip('[').strip(']')).strip('\"')
	index_name = ['id_0']
	for i in range(1, 240):
		index_name.append('id_'+str(i))
	#print(index_name)
	final_df = pd.DataFrame(index_name, columns=['id'])
	final_df['value'] = pd.DataFrame(test_pred)
	#index_name = index_name.append(test_pred)
	#final_df = pd.DataFrame(test_pred, columns=['id','value'])
	#final_df.index.name(index_name)
	#final_df = final_df.rename(columns = ['id,value'])
	#print(final_df)
	final_df.to_csv('output.csv', index=False)

test()

