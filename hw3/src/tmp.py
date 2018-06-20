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

parser = argparse.ArgumentParser(prog='plot_saliency.py', description='ML-Assignment3 visualize attention heat map.')
parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=1)
args = parser.parse_args()
#model_name = "model-%s.h5" %str(args.epoch)
#model_path = os.path.join(model_dir, model_name)
emotion_classifier = load_model('../my_models/asgard_model_500')
#print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))
private_pixels, Y_train = read_data()
#private_pixels = load_pickle('fer2013/test_with_ans_pixels.pkl')
#private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) for i in range(len(private_pixels)) ]
input_img = emotion_classifier.input
img_ids = [1]
idx=1

val_proba = emotion_classifier.predict(private_pixels[idx].reshape((1,48,48,1)))
pred = val_proba.argmax(axis=-1)
target = K.mean(emotion_classifier.output[:, pred])
grads = K.gradients(target, input_img)[0]
fn = K.function([input_img, K.learning_phase()], [grads])
heatmap = fn([private_pixels, False])
heatmap = heatmap.reshape(48,48)
