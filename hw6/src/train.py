import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.layers import Dense, Input
from keras.models import Model, load_model
from scipy import spatial
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))

img_path = sys.argv[1]
model_path = sys.argv[2]

if 'session' in locals() and session is not None:
	print('Close interactive session')
	session.close()

def main():
	X = np.load(img_path)
	X = X/255


	input_img = Input(shape=(784,))
	encoded = Dense(512, activation='relu')(input_img)
	encoded = Dense(128, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)

	#decoded = Dense(128, activation='relu')(encoded)
	decoded = Dense(512, activation='relu')(encoded)
	decoded = Dense(784)(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.summary()
	encoder = Model(input_img, encoded)
	encoder.summary()

	encoded_input = Input(shape=(32,))
	#decoder_layer = autoencoder.layers[-1]
	
	#decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adam', loss='mse')
	autoencoder.fit(X, X, epochs=400, batch_size=512, shuffle=True, validation_split=0.1)

	encoder.save(model_path)
	encoder = load_model(model_path)
	encoded_imgs = encoder.predict(X)

if __name__ == '__main__':
	main()
