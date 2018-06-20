from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt

def read_data():
	width = height = 48
	with open('../data/train.csv', 'r') as f:
		data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
		data = np.array(data)
		X_train = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
		Y_train = data[::width*height+1].astype('int')
	#Y_train = to_categorical(Y_train, 7)
	X_train /= 255

	return X_train[-7000:], Y_train[-7000:]

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
		color="white" if cm[i, j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')


def main():
	X_val, Y_val = read_data()
	emotion_classifier = load_model('../asgard_model_500')
	np.set_printoptions(precision=2)
	predictions = emotion_classifier.predict(X_val)
	predictions = predictions.argmax(axis=-1)
	conf_mat = confusion_matrix(predictions, Y_val)

	plt.figure()
	plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	plt.savefig('../confusion_matrix.png')
	plt.show()


if __name__ == '__main__':
	main()
