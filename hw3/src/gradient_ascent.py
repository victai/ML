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
	emotion_classifier = load_model('../my_models/asgard_model_500')
	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
	input_img = emotion_classifier.input
	print(layer_dict)
	name_ls = ['conv2d_2']
	collect_layers = list()
	collect_layers.append(layer_dict[name_ls[0]].output)
	print(collect_layers)
	for cnt, c in enumerate(collect_layers):
		#filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
		filter_imgs = []
		nb_filter = 32
		#nb_filter = c.shape[-1]
		for filter_idx in range(nb_filter):
			input_img_data = np.random.random((1, 48, 48, 1))
			loss = K.mean(c[:,:,:,filter_idx])
			grads = normalize(K.gradients(loss,input_img)[0])
			iterate = K.function([input_img, K.learning_phase()],[loss,grads])

			filter_imgs.append(grad_ascent(NUM_STEPS, input_img_data, iterate))

		for it in range(NUM_STEPS//RECORD_FREQ):
			fig = plt.figure(figsize=(14,8))
			for i in range(nb_filter):
				ax = fig.add_subplot(int(nb_filter)/8,8,i+1)
				raw_img = filter_imgs[i][it][0].squeeze()
				ax.imshow(deprocess_image(raw_img),cmap='GnBu')
				plt.xticks(np.array([]))
				plt.yticks(np.array([]))
				plt.xlabel('{:.3f}'.format(filter_imgs[i][it][1]))
				plt.tight_layout()
			fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],it*RECORD_FREQ))
			fig.savefig('e{}'.format(it*RECORD_FREQ))
	'''
	'''
if __name__ == '__main__':
	main()
