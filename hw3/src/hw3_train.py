import numpy as np
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
#np.set_printoptions(threshold=np.inf)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def CNN(X_train, Y_train):
	X_val = X_train[-7000:]
	Y_val = Y_train[-7000:]
	X_train = X_train[:-7000]
	Y_train = Y_train[:-7000]
	X_train = np.vstack((X_train, X_train, X_train))
	Y_train = np.vstack((Y_train, Y_train, Y_train))

	model = Sequential()
	model.add(Convolution2D(32,(3,3), input_shape=(48,48,1), padding='same'))
	model.add(BatchNormalization(input_shape=(48,48,1)))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(64,(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.2))

	model.add(Convolution2D(128,(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.3))

	model.add(Convolution2D(256,(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.4))
	#model.add(Dense(512))
	#model.add(LeakyReLU(alpha=0.1))
	#model.add(Dropout(0.3))
	model.add(Dense(7, activation='softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False,	  # randomly flip images
		zoom_range=0.2
	)
	
	datagen.fit(X_train)
	model.fit_generator(
			datagen.flow(X_train, Y_train, batch_size=64),
			steps_per_epoch=round(X_train.shape[0]/64),
			epochs=500,
			validation_data=(X_val, Y_val),
			workers=4
	)
	
	#model.fit(X_train, Y_train, batch_size=100, epochs=50, validation_split=0.1, shuffle=True)
	score = model.evaluate(X_train, Y_train)
	print('Total loss: ', score[0])
	print('Accuracy: ', score[1])
	model.save('model_1')

def Read_data():
	width = height = 48
	with open(sys.argv[1], 'r') as f:
		data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
		data = np.array(data)
		X_train = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
		Y_train = data[::width*height+1].astype('int')
	Y_train = to_categorical(Y_train, 7)
	X_train /= 255
	'''
	Train = pd.read_csv(sys.argv[1]).values.astype(str)
	X_train = Train[:, 1]
	print(X_train.shape)
	Y_train = Train[:, 0]
	#X_train = X_train.flatten().reshape(-1,48,48)
	X_train = np.core.defchararray.split(X_train, sep=' ')
	print(X_train.shape)
	print(X_train)
	for i in range(1, 10):
		a = X_train[i]
		a = a.split()
		tmp = np.vstack((tmp, np.array(a, int).flatten().reshape(48,48)))
	tmp.reshape(10, 48, 48)
	print(tmp)
	print(tmp.shape)
	'''
	
	return X_train, Y_train

if __name__ == '__main__':
	X_train, Y_train = Read_data()
	CNN(X_train, Y_train)
