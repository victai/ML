import jieba
import sys
from util import *
from gensim.models import Word2Vec
from scipy import spatial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--no_stopwords', type=int)
parser.add_argument('--stopword_file', type=str)
parser.add_argument('--size', type=int)
parser.add_argument('--w2v_model', type=str)
args = parser.parse_args()

jieba.set_dictionary('data/dict.txt.big')

stop = []
if args.no_stopwords == 1:
	with open(args.stopword_file, 'r', encoding='utf8') as fp:
		for line in fp:
			stop.append(line.replace('\n',''))

def read_data(path):
	q = []
	a = []
	with open(path,'r',encoding='utf8') as fp:
		fp.readline()
		for line in fp:
			#print(line)
			wq = []
			wa = []
			tmp = line.replace('\n','').split(',')
			tmpq = tmp[1].replace('\t','')[2:]
			tmpa = tmp[2].split('\t')
			words = jieba.cut(tmpq.replace(' ',''), cut_all=False)
			for word in words:
				if word not in stop:
					wq.append(word)
			q.append(wq)
			for word in tmpa:
				waa = []
				t = word[2:]
				words = jieba.cut(t.replace(' ',''), cut_all=False)
				for w in words:
					if w not in stop:
						waa.append(w)
				a.append(waa)

	return q, a

q, a = read_data(args.data)

model = Word2Vec.load(args.w2v_model)
transformByWord2Vec(q, model, args.size)
transformByWord2Vec(a, model, args.size)

q = np.asarray(q)
a = np.asarray(a)

for i in range(5060):
	q[i] = np.asarray(q[i])
	num = q[i].shape[0]
	q[i] = np.sum(q[i],axis=0)
	q[i] = q[i] / num

for i in range(5060*6):
	a[i] = np.asarray(a[i])
	num = a[i].shape[0]
	a[i] = np.sum(a[i],axis=0)
	a[i] = a[i] / num

with open(args.output, 'w') as fp:
	fp.write('id,ans\n')
	for i in range(5060):
		qr = q[i]
		ans = -1
		m = 0
		for j in range(6):
			ar = a[i*6+j]
			d = spatial.distance.cosine(qr, ar)
			sim = 1 - d
			if sim > m:
				ans = j
				m = sim
		fp.write('%d,%d\n' %(i+1, ans))
