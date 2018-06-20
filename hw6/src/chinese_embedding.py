import sys
import jieba
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from adjustText import adjust_text
import ipdb

font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
prop = matplotlib.font_manager.FontProperties(fname=font_path)

input_file = sys.argv[1]
output_pic = sys.argv[2]
model_path = sys.argv[3]
jieba.set_dictionary('../jieba/extra_dict/dict.txt.big')

def train_w2v():
	with open(input_file, 'r') as f:
		data = f.read().split('\n')[:-1]
	for i, line in enumerate(data):
		data[i] = data[i].split()
	model = Word2Vec(data, size=300, min_count=5000, workers=4, iter=50)
	model.save(model_path)
	

def main():
	train_w2v()
	model = Word2Vec.load(model_path)
	words = []
	weights = []
	for word in model.wv.vocab:
		words.append(word)
		weights.append(model.wv[word])

	tsne = TSNE(n_components=2)
	weights = tsne.fit_transform(weights)
	
	plt.figure(figsize=(10, 10))
	plt.scatter(weights[:,0], weights[:,1], marker='.')
	texts = []
	for i in range(len(words)):
		texts.append(plt.text(weights[i,0], weights[i,1], words[i], fontproperties=prop))
	plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))+' iterations')
	plt.savefig(output_pic)
	plt.show()

if __name__ == '__main__': 
	main()
