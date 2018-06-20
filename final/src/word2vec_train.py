import sys
from util import *
from gensim.models import Word2Vec
from gensim.models import word2vec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--w2v_model', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--sg', type=int)
parser.add_argument('--size', type=int)
args = parser.parse_args()

data = []
#read
'''
with open('train_merge.txt', 'r', encoding='utf8') as fp:
	for line in fp:
		data.append(line.replace('\n','').split(' '))
'''

sentence = word2vec.Text8Corpus(args.data)
model = Word2Vec(sentence, min_count=1, size=args.size, sg=args.sg, iter=20)
model.save(args.w2v_model)
