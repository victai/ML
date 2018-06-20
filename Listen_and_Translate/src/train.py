import tensorflow as tf
import numpy as np
import pickle
import argparse
import ipdb
from keras.layers import LSTM, Input, Dense, Embedding, merge
from keras.models import Model
from gensim.models import Word2Vec

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

num_encoder_tokens=200	#word2vec dimension
num_vocab=2449			#2446 + <unk>(0) + <BOS>(1) + <EOS>(2)
train_data_size=45036
max_len=30
dictionary={}
reverse_dict={}

def read_data(args):
	with open(args.training_data, 'rb') as f:
		data = pickle.load(f)
	with open(args.training_caption, 'r') as f:
		caption = f.read().split('\n')[:-1]
	for i, line in enumerate(caption):
		caption[i] = caption[i].split()
	with open('data/test.csv', 'r') as f:
		test_caption = f.read().split('\n')[:-1]
	with open('data/test.data', 'rb') as f:
		test_data = pickle.load(f)
	for i, line in enumerate(test_caption):
		test_caption[i] = test_caption[i].split(',')
		for j in range(4):
			test_caption[i][j] = test_caption[i][j].split()

	#all_caption = caption + test_caption
	#model = Word2Vec(all_caption, size=200, workers=4, min_count=1)
	#model.save('models/word2vec_model.h5')
	word2vec_model = Word2Vec.load('models/word2vec_model.h5')
	'''
	dictionary = {}
	weights = np.zeros((2449, 200))
	for i, vocab in enumerate(word2vec_model.wv.vocab):
		dictionary[vocab] = i+3
		weights[i+3] = word2vec_model.wv[vocab]
	'''
	global dictionary, reverse_dict
	with open('weights.pkl', 'rb') as f:
		weights = pickle.load(f)
	with open('dictionary.pkl', 'rb') as f:
		dictionary = pickle.load(f)
	for i, word in enumerate(dictionary):
		reverse_dict[dictionary[word]] = word

	for i, line in enumerate(data):
		data[i] = np.array(data[i])
		data[i] = np.vstack((data[i], np.zeros((250-len(data[i]),39))))
	for i, line in enumerate(test_data):
		test_data[i] = np.vstack((test_data[i], np.zeros((250-len(test_data[i]),39))))
	
	for i, line in enumerate(caption):
		caption[i] = ['<BOS>'] + caption[i] + ['<EOS>']
		for j in range(30):
			if j >= len(caption[i]):	caption[i].append([0 for k in range(200)])
			elif caption[i][j] in word2vec_model.wv.vocab:
				caption[i][j] = weights[dictionary[caption[i][j]]]
			elif caption[i][j] == '<BOS>':
				caption[i][j] = [0 for k in range(200)]
			elif caption[i][j] == '<EOS>':
				caption[i][j] = [0 for k in range(200)]
			else:	caption[i][j] = [0 for k in range(200)]

	for i, line in enumerate(test_caption):
		for j, options in enumerate(test_caption[i]):
			test_caption[i][j] = ['<BOS>'] + test_caption[i][j] + ['<EOS>']
			for k in range(30):
				if k >= len(test_caption[i][j]): test_caption[i][j].append([0 for l in range(200)])
				elif test_caption[i][j][k] in word2vec_model.wv.vocab:
					test_caption[i][j][k] = weights[dictionary[test_caption[i][j][k]]]
				else:   test_caption[i][j][k] = [0 for l in range(200)]

	print(len(data))
	print(len(caption))
	
	return np.array(data), np.array(caption), weights, \
			np.array(test_data), np.array(test_caption)
'''
def decode_sequence(input_seq, encoder_model, decoder_model, caption_test):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_vocab))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, dictionary['<BOS>']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_dict[sampled_token_index]
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
'''

def distance(arr1, arr2):
	return	np.linalg.norm(arr1-arr2)

def main(args):
	audio_train, caption_train, weights, audio_test, caption_test = read_data(args)
	print(caption_train.shape)
	decoder_target_data = caption_train[:, 1:, :]
	decoder_input_data = caption_train[:, :-1, :]

	encoder_inputs = Input(shape=(None, 39))
	forward_encoder = LSTM(100, return_state=True, return_sequences=True)
	backward_encoder = LSTM(100, return_state=True, return_sequences=True, go_backwards=True)

	forward, forward_h, forward_c = forward_encoder(encoder_inputs)
	backward, backward_h, backward_c = backward_encoder(encoder_inputs)
	merged = merge([forward, backward], mode='sum')

	encoder = LSTM(100, return_state=True)

	encoder_outputs, state_h, state_c = encoder(merged)
	#encoder_outputs_att = AttentionWithContext()(encoder_outputs)
	encoder_states = [state_h, state_c]
	
	decoder_inputs = Input(shape=(None,200))
	#decoder_inputs = Embedding(input_dim=num_vocab,
	#							output_dim=200,
	#							input_length=29,
	#							weights=[weights],
	#							trainable=False)(decoder_inputs)

	decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	
	decoder_dense = Dense(200, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	model.compile(optimizer='rmsprop', loss='mse')
	model.fit([audio_train, decoder_input_data],
			decoder_target_data,
			batch_size=512,
			epochs=3,
			validation_split=0.1)
	model.summary()

	#model.save('models/my_model.h5')
	##############################################################
	
	with open('prediction.csv', 'w') as f:
		f.write('id,answer\n')
		for i in range(len(audio_test)):
			m = 1e10
			ans = -1
			t = audio_test[i].reshape(1,250,39)
			for j in range(4):
				option = caption_test[i][j].reshape(1,30,200)
				Y = model.predict([t, option])
				d = distance(Y, option)
				if d < m:
					ans = j
					m = d
			f.write('%d,%d\n'%(i+1,ans))

		
	'''
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

		decoded_sentence = decode_sequence(audio_test[i],
											encoder_model,
											decoder_model,
											caption_test)
		print(decoded_sentence)
	'''
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_data', type=str)
	parser.add_argument('--training_caption', type=str)
	parser.add_argument('--model_path', type=str)
	args = parser.parse_args()
	main(args)
