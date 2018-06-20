from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pickle
import numpy as np

my_category = ["Animation|Children's|Adventure", "Crime|Mystery", "Thriller|Horror", "Romance|Drama|Musical", "War|Western|Action", "Documentary|Film-Noir", "Sci-Fi|Fantasy|Comedy"]

def read_data():
	with open('data/movies.csv', 'r', errors='ignore') as f:
		movies = f.read().split('\n')[1:-1]
		print(movies[0])
	with open('models/weight.pkl', 'rb') as f:
		weights = pickle.load(f)
	with open('data/genres.txt', 'r') as f:
		g = f.read().split('\n')
	genres_id = {}
	for i in range(18):
		genres_id[g[i]] = i
	genres = {}
	'''
	for i in range(len(movies)):
		movie_category = movies[i].split('::')[2].split('|')[0]
		if movie_category not in genres.keys():
			genres[movie_category] = [i]
		else:
			genres[movie_category].append(i)
	'''
	for i in range(len(movies)):
		movie_category = movies[i].split('::')[2].split('|')[0]
		for cat in my_category:
			c = [i for i in cat.split('|')]
			if movie_category in c:
				if cat not in genres.keys():
					genres[cat] = [i]
				else:
					genres[cat].append(i)

	return weights, genres

def main():
	weights, genres = read_data()
	tsne = TSNE(n_components=2, random_state=0)
	weights = tsne.fit_transform(weights)
	plt.figure(figsize=(10,10))
	for key in genres.keys():
		genres[key] = np.array(weights[genres[key]])
		x = genres[key][:,0]
		y = genres[key][:,1]
		plt.scatter(x, y, marker='.', label=key)
	plt.legend(loc='upper right')
	plt.savefig('tsne.png')
	plt.show()
	

	print(genres[my_category[0]])

if __name__ == '__main__':
	main()
