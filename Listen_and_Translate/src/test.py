import tensorflow as tf
import numpy as np
import pickle
import argparse
import ipdb
from keras.layers import LSTM, Input, Dense, Embedding
from keras.models import Model
from gensim.models import Word2Vec

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def read_data():
	with open('data/test.data', 'rb') as f:
		data = pickle.load(f)
	with open('data/test.csv', 'r') as f:
		caption = f.read().split('\n')[:-1]

	for i, line in enumerate(caption):
		caption[i] = caption[i].split(',')
		for j in range(4):
			caption[i][j] = caption[i][j].split()

	word2vec_model = Word2Vec.load('models/word2vec_model.h5')
	
	for i, line in enumerate(data):
		data[i] = np.vstack((data[i], np.zeros((250-len(data[i]),39))))
	for i, line in enumerate(caption):
		for j, options in enumerate(caption[i]):
			caption[i][j] = ['<BOS>'] + caption[i][j] + ['<EOS>']
			for k in range(30):
				if k >= len(caption[i][j]):	caption[i][j].append([0 for k in range(200)])
				elif caption[i][j][k] in word2vec_model.wv.vocab:
					caption[i][j][k] = list(word2vec_model.wv[caption[i][j][k]])
				else:	caption[i][j][k] = [0 for k in range(200)]
	ipdb.set_trace()

	print(len(data))
	print(len(caption))
	return np.array(data), np.array(caption)


def decode_sequence(input_seq, encoder_model, decoder_model):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, target_token_index['\t']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char
		# Exit condition: either hit max length # or find stop character.
		if (sampled_char == '\n' or
			len(decoded_sentence) > max_decoder_seq_length):
			stop_condition = True

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1, num_decoder_tokens))
			target_seq[0, 0, sampled_token_index] = 1.
			# Update states
			states_value = [h, c]

	return decoded_sentence

def main():
	data, caption = read_data()

	encoder_model = Model(encoder_inputs, encoder_states)

	decoder_state_input_h = Input(shape=(None, 200))
	decoder_state_input_c = Input(shape=(None, 200))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(
					decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model(
				[decoder_inputs] + decoder_states_inputs,
				[decoder_outputs] + decoder_states)

	decode_sequence(data[0], encoder_model, decoder_model)

	return

if __name__ == '__main__':
	main()
