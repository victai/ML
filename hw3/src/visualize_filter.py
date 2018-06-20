import os
import sys
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from termcolor import colored,cprint
import numpy as np
from utils import *

nb_class = 7
LR_RATE = 1e-2
NUM_STEPS = 200
RECORD_FREQ = 10

cnt = 0

def read_data():
	width = height = 48
	with open('../data/train.csv', 'r') as f:
		data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
		data = np.array(data)
		X_train = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
		Y_train = data[::width*height+1].astype('int')
	X_train /= 255
	return X_train, Y_train

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x += 0.5
	x = np.clip(x, 0, 1)
	x *= 255
	return x

def normalize(x):
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_ascent(num_step,input_image_data,iterate):
	filter_images = []
	global cnt
	print(cnt)
	for epochs in range(NUM_STEPS):
		loss, grad = iterate([input_image_data, False])
		input_image_data += LR_RATE * grad 
		if epochs % RECORD_FREQ == 0:
			filter_images.append((input_image_data, loss))
			print('#{}, loss rate: {}'.format(epochs, loss))
	cnt += 1
	return filter_images

def main():
	X_train, Y_train = read_data()
	emotion_classifier = load_model('../my_models/asgard_model_500')
	img_id = 10000
	photo = X_train[img_id]
	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
	input_img = emotion_classifier.input
	print(layer_dict)
	name_ls = ['conv2d_2']
	collect_layers = list()
	collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict[name_ls[0]].output]))

	for cnt, fn in enumerate(collect_layers):
		im = fn([photo.reshape(1,48,48,1), False])
		fig = plt.figure(figsize=(14,8))
		nb_filter = 32
		for i in range(nb_filter):
			ax = fig.add_subplot(nb_filter/8,8,i+1)
			ax.imshow(im[0][0,:,:,i],cmap='GnBu')
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.tight_layout()
		fig.suptitle('Output of layer{} (Given image{})'.format(cnt,img_id))
		fig.savefig('layer{}'.format(cnt))

if __name__ == '__main__':
	main()
