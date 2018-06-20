import numpy as np
import pickle
import ipdb
import argparse
from keras.layers import LSTM, Input, Dense, Embedding
from keras.models import Model
from gensim.models import Word2Vec

num_encoder_tokens=250
num_decoder_tokens=30

def read_data(args):
	with open(args.training_data, 'rb') as f:
		data = pickle.load(f)
	with open(args.training_caption, 'r') as f:
		caption = f.read().split('\n')
	for i, line in enumerate(caption):
		caption[i] = caption[i].split()
	with open('data/test.csv', 'r') as f:
		test_caption = f.read().split('\n')
	for i, line in enumerate(test_caption):
		test_caption[i] = test_caption[i].replace(',', ' ').split()
	all_caption = caption + test_caption
	#model = Word2Vec(all_caption, size=200, workers=4)
	#model.save('models/word2vec_model.h5')
	word2vec_model = Word2Vec.load('models/word2vec_model.h5')
	'''
	dictionary = {}
	weights = np.zeros((1645, 200))
	print(len(word2vec_model.wv.vocab))
	for i, vocab in enumerate(word2vec_model.wv.vocab):
		dictionary[vocab] = i+1
		weights[i+1] = word2vec_model.wv[vocab]
	'''
	with open('weights.pkl', 'rb') as f:
		weights = pickle.load(f)
	with open('dictionary.pkl', 'rb') as f:
		dictionary = pickle.load(f)
	
	for i, line in enumerate(data):
		data[i] = np.array(data[i])
		data[i] = np.vstack((data[i], np.zeros((250-len(data[i]),39))))
	for i, line in enumerate(caption):
		caption[i] = ['<s>'] + caption[i]
		for j in range(30):
			if j >= len(caption[i]):	caption[i].append(0)
			elif caption[i][j] in word2vec_model.wv.vocab:
				caption[i][j] = dictionary[caption[i][j]]
			else:	caption[i][j] = 0
	
	return np.array(data), caption, weights, word2vec_model

	
def main(args):
	audio_train, caption_train, weights, word2vec_model = read_data(args)
	decoder_target_data = [None]*len(caption_train)
	for i in range(len(caption_train)):
		decoder_target_data[i] = caption_train[i][1:] + [0]
		for j in range(30):
			decoder_target_data[i][j] = weights[decoder_target_data[i][j]]

	decoder_target_data = np.array(decoder_target_data)
	caption_train = np.array(caption_train)
	
	encoder_inputs = Input(shape=(1,250,39))
	encoder = LSTM(100, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	encoder_states = [state_h, state_c]

	decoder_inputs = Input(shape=(1,))
	decoder_inputs = Embedding(input_dim=1645,
								output_dim=200,
								input_length=30,
								weights=[weights],
								trainable=False)(decoder_inputs)

	decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	
	decoder_dense = Dense(num_decoder_tokens, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	model.fit([audio_train, caption_train],
			decoder_target_data,
			batch_size=512,
			epochs=3,
			validation_split=0.2)
				

	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_data', type=str)
	parser.add_argument('--training_caption', type=str)
	parser.add_argument('--model_path', type=str)
	args = parser.parse_args()
	main(args)
