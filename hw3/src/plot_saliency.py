import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def read_data():
	width = height = 48
	with open('../data/train.csv', 'r') as f:
		data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
		data = np.array(data)
		X_train = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
		Y_train = data[::width*height+1].astype('int')
	#Y_train = to_categorical(Y_train, 7)
	X_train /= 255
	return X_train, Y_train

def main():
	emotion_classifier = load_model('../my_models/asgard_model_500')
	private_pixels, Y_train = read_data()
	input_img = emotion_classifier.input
	img_ids = [388, 6020, 10000]

	for idx in img_ids:
		pxl = private_pixels[idx].reshape((1,48,48,1))
	    val_proba = emotion_classifier.predict(pxl)
	    pred = val_proba.argmax(axis=-1)
	    target = K.mean(emotion_classifier.output[:, pred])
	    grads = K.gradients(target, input_img)[0]
	    grads = (grads-K.mean(grads))/K.std(grads)
	    grads = (grads-K.min(grads))/(K.max(grads)-K.min(grads))

		fn = K.function([input_img, K.learning_phase()], [grads])
		print(pred)

	    thres = 0.5
	    see = pxl.reshape(48, 48)
	    #for i in range(48):
	    #    for j in range(48):
	    #        print(heatmap[i][j])
	    see[np.where(heatmap <= thres)] = np.mean(see)
	    plt.figure()
		plt.imshow(pxl.reshape(48,48), cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.show()
		plt.savefig(str(idx) + 'original.png')
		plt.clf()

		plt.figure()
		plt.imshow(heatmap,cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.show()
		fig.savefig(str(idx) + 'heatmap.png')
		plt.clf()
										    
		plt.figure()
		plt.imshow(see, cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.show()
		fig.savefig(str(idx) + 'mask.png')
		plt.clf()

if __name__ == '__main__':
	main()
