import sys
import numpy as np
import skimage.io, skimage.transform
import os
import ipdb

pic_dir = sys.argv[1]
pic_id = sys.argv[2]

size = 600

def read_all():
	pic_files = os.listdir(pic_dir)
	pics = np.zeros((len(pic_files), size*size*3))
	for i, pic in enumerate(pic_files):
		pic2 = os.path.join(pic_dir, pic)
		tmp = skimage.io.imread(pic2)
		#tmp = skimage.transform.resize(tmp, (size,size,3))
		#skimage.io.imsave('resize/%s' %(pic), tmp)
		pics[i] = tmp.reshape(1, -1)
	return pics

def reconstruct(U, mu_pics, k):
	path = os.path.join(pic_dir, pic_id)
	pic = skimage.io.imread(path).reshape(1,-1)
	pic = pic - mu_pics

	_U = np.zeros((k, size*size*3))
	for i in range(k):
		_U[i] = U.T[i]
	weights = _U.dot(pic.T)

	new_pic = np.zeros(size*size*3)
	for i in range(k):
		new_pic += _U[i]*weights[i]
	new_pic += mu_pics
	new_pic = trans(new_pic)
	new_pic = new_pic.reshape(size,size,3).astype(np.uint8)
	skimage.io.imsave('reconstruct.jpg', new_pic, quality=100)

def draw_average(pics):
	mu_pics = np.around(sum(pics)/pics.shape[0]).astype(int)
	average_face = mu_pics.reshape(size,size,3)
	skimage.io.imsave('resize_average.jpg', average_face, quality=100)

def draw_eigenface(U, k):
	for i in range(k):
		tmp = U.T[i].reshape(size,size,3)
		eig_pic = trans(tmp)
		skimage.io.imsave('eigen_%d.jpg' %(i), eig_pic, quality=100)

def trans(pic):
	pic -= np.min(pic)
	pic /= np.max(pic)
	pic = (pic*255).astype(np.uint8)
	return pic

def main():
	pics = read_all()
	#draw_average(pics)

	mu_pics = pics.mean(axis=0)
	U,s,V = np.linalg.svd((pics-mu_pics).T, full_matrices=False)
	'''
	s_sum = sum(s)
	for i in range(4):
		print(s[i]/s_sum)
	'''	
	#draw_eigenface(U, 4)
	
	reconstruct(U, mu_pics, 4)


if __name__ == '__main__':
	main()
