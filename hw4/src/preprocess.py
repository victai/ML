import numpy as np
import argparse
import pickle
from gensim import corpora
from gensim.models import Word2Vec

def train_word2vec(args):
	with open(args.labeled_path, 'r') as f:
		labeled_data = f.read().split('\n')[:-1]
	with open(args.test_path, 'r') as f:
		test_data = f.read().split('\n')[1:-1]
	with open(args.unlabeled_path, 'r') as f:
		unlabeled_data = f.read().split('\n')[:-1]

	X_train = [None]*200000
	Y_train = np.zeros(200000, dtype=int)
	X_test = [None]*200000
	for i, line in enumerate(labeled_data):
		X_train[i] = line[10:]
		Y_train[i] = int(line[0])

	for i, line in enumerate(test_data):
		X_test[i] = line.split(',', 1)[1]

	for i in range(200000):
		X_train[i] = X_train[i].split()
		X_test[i] = X_test[i].split()
	for i in range(len(unlabeled_data)):
		unlabeled_data[i] = unlabeled_data[i].split()

	all_data = X_train + X_test + unlabeled_data

	model = Word2Vec(all_data, size=100, min_count=5, workers=4)
	model.save(args.word2vec_model)

	return

def make_dictionary_weights(args):
	vec_model = Word2Vec.load(args.word2vec_model)
	weights = np.zeros((55777, 100))
	dictionary = {}
	for i, vocab in enumerate(vec_model.wv.vocab):
		weights[i+1] = vec_model.wv[vocab]
		dictionary[vocab] = i+1
	with open('weights.pkl', 'wb') as f:
		pickle.dump(weights, f)
	with open('dictionary.pkl', 'wb') as f:
		pickle.dump(dictionary, f)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--labeled_path', type=str)
	parser.add_argument('--unlabeled_path', type=str)
	parser.add_argument('--test_path', type=str)
	parser.add_argument('--word2vec_model', type=str)
	args = parser.parse_args()
	#train_word2vec(args)
	make_dictionary_weights(args)
	
if __name__ == '__main__':
	main()
