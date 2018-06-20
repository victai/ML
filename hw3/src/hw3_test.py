import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
#np.set_printoptions(threshold=np.inf)

def Test(X_test):
	model = load_model(sys.argv[3])
	Y = model.predict(X_test)
	result = np.zeros(X_test.shape[0], dtype=int)
	result = np.argmax(Y, axis=1)
	idx = np.arange(X_test.shape[0], dtype=int)
	idx = np.vstack((idx,result))
	df = pd.DataFrame(idx.T, columns=['id', 'label'])
	df.to_csv(sys.argv[2], index=False)

def Read_data():
	width = height = 48
	with open(sys.argv[1], 'r') as f:
		data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
		data = np.array(data)
		X_test = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
	X_test /= 255
	
	return X_test

if __name__ == '__main__':
	X_test = Read_data()
	Test(X_test)
