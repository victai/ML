import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.layers import Dense, Input
from keras.models import Model, load_model
from scipy import spatial
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))

img_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]
model_path = sys.argv[4]

if 'session' in locals() and session is not None:
	print('Close interactive session')
	session.close()

def test(kmeans):
	with open(testing_data_path, 'r') as f:
		testing_data = f.read().split('\n')[1:-1]
	
	ans = np.zeros(len(testing_data), dtype=int)
	for line in testing_data:
		line = line.split(',')
		idx = int(line[0]); x = int(line[1]); y = int(line[2])
		x_label = kmeans.labels_[x]
		y_label = kmeans.labels_[y]
		if x_label == y_label:
			ans[idx] = 1
		#elif (x_label in [0,1,2]) and (y_label in [0,1,2]):
		#	ans[idx] = 1
		#elif (x_label in [1,3]) and (y_label in [1,3]):
		#	ans[idx] = 1

	with open(output_path, 'w') as f:
		f.write('ID,Ans\n')
		for i in range(len(testing_data)):
			f.write('%d,%d\n' %(i, ans[i]))

def visualization(encoder):
	data = np.load('visualization.npy')
	encoded_imgs = encoder.predict(data)

	X_embedded = TSNE(n_components=2).fit_transform(encoded_imgs)
	with open('X_embedded.pkl', 'wb') as f:
		pickle.dump(X_embedded, f)
	kmeans = KMeans(n_clusters=2)
	kmeans.fit(encoded_imgs)
	label_0 = kmeans.labels_ == 0
	label_1 = kmeans.labels_ == 1
	with open('label_0.pkl', 'wb') as f:
		pickle.dump(label_0, f)
	with open('label_1.pkl', 'wb') as f:
		pickle.dump(label_1, f)
	'''
	plt.scatter(X_embedded[label_0, 0], X_embedded[label_0, 1], c='b', label='dataset A', s = 0.2)
	plt.scatter(X_embedded[label_1, 0], X_embedded[label_1, 1], c='r', label='dataset A', s = 0.2)
	plt.legend()
	plt.savefig('my_prediction.png')
	plt.close()

	plt.scatter(X_embedded[:5000, 0], X_embedded[:5000, 1], c='b', label='dataset A', s = 0.2)
	plt.scatter(X_embedded[5000:, 0], X_embedded[5000:, 1], c='r', label='dataset A', s = 0.2)
	plt.legend()
	plt.savefig('true_label.png')
	'''

def main():
	X = np.load(img_path)
	X = X/255

	encoder = load_model(model_path)
	encoded_imgs = encoder.predict(X)

	kmeans = KMeans(n_clusters=2)
	kmeans.fit(encoded_imgs)
	test(kmeans)
	
	#visualization(encoder)

if __name__ == '__main__':
	main()
