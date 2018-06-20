import numpy as np
import tensorflow as tf
import pickle
import argparse
from gensim.models import Word2Vec
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, Embedding, Flatten
from keras.optimizers import Adam

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def Test(args):
	model = load_model(args.model)
	with open('data/testing_data.txt', 'r') as f:
		data = f.read().split('\n')[1:-1]
	X_test = [None]*200000
	for i, line in enumerate(data):
		X_test[i] = line.split(',', 1)[1].split()

	vec_model = Word2Vec.load(args.word2vec_model)
	with open('dictionary.pkl', 'rb') as f:
		dictionary = pickle.load(f)
	for i in range(200000):
		for j in range(39):
			l = len(X_test[i])
			if j >= l:
				X_test[i].append(0)
			elif X_test[i][j] in vec_model.wv.vocab:
				X_test[i][j] = dictionary[X_test[i][j]]
			else:
				X_test[i][j] = 0 

	Y = model.predict(X_test)
	idx = np.arange(200000)
	ans = np.zeros(200000, dtype=int)
	for i in range(200000):
		if Y[i] > 0.5:	ans[i] = 1
	with open(args.output, 'w') as f:
		f.write('id,label\n')
		for i in range(200000):
			f.write(str(i) + ',' + str(ans[i]) + '\n')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--word2vec_model', type=str)
	parser.add_argument('--test_data', type=str)
	parser.add_argument('--output', type=str)
	args = parser.parse_args()
	Test(args)

if __name__ == '__main__':
	main()
