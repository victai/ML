import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad
np.set_printoptions(threshold=np.inf)

dim = 116

sigmoid = lambda z : 1/(1+ np.exp(-z))

def Read_data():
	X_train = pd.read_csv(sys.argv[1], encoding='Big5').values
	Y_train = pd.read_csv(sys.argv[2], encoding='Big5').values
	X_train = np.hstack((X_train, X_train[:, :5]**2))
	X_train = np.hstack((X_train, X_train[:, :5]**3))
	#X_train = np.hstack((X_train, X_train[:, :5]**4))
	#X_train = np.hstack((X_train, X_train[:, :5]**5))
	#X_train = np.hstack((X_train, X_train[:, :5]**6))
	X_test = pd.read_csv('data/X_test', encoding='Big5').values
	X_test = np.hstack((X_test, X_test[:, :5]**2))
	X_test = np.hstack((X_test, X_test[:, :5]**3))
	#X_test = np.hstack((X_test, X_test[:, :5]**4))
	#X_test = np.hstack((X_test, X_test[:, :5]**5))
	#X_test = np.hstack((X_test, X_test[:, :5]**6))
	X_all = np.concatenate((X_train, X_test))
	mu = sum(X_all) / X_all.shape[0]
	sigma = np.std(X_all, axis=0)
	return (X_train-mu)/sigma, Y_train, (X_test-mu)/sigma

def Test(model, X_test):
	result = model.predict(X_test, batch_size=100)
	index_name = [i for i in range(1,X_test.shape[0]+1)]
	df = pd.DataFrame(index_name, columns=['id'])
	df['label'] = pd.DataFrame(result.astype(int))
	df.to_csv('output.csv', index=False)

if __name__ == '__main__':
	X_train, Y_train, X_test= Read_data()
	model = Sequential()
	model.add(Dense(input_dim=dim, units=100, activation='relu'))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.2))
	#model.add(Dense(units=100, activation='relu'))
	#model.add(Dropout(0.2))
	#model.add(Dense(units=100, activation='relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=0.01), metrics=['accuracy'])
	model.fit(X_train, Y_train, batch_size=100, epochs=50, validation_split=0.1, shuffle=True)
	score = model.evaluate(X_train, Y_train)
	print('Total loss: ', score[0])
	print('Accuracy: ', score[1])
	Test(model, X_test)
