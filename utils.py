import numpy as np
import gzip
from ops import *

def extract_data(filename, num_images, IMAGE_WIDTH):
	print("Extracting: ", filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
		return data

def extract_labels(filename, num_images):
	print("Extracting: ", filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		return labels

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, stride_c = 1, pool = 2, stride_p = 2):
	# forward propagation
	conv1 = convolution(image, f1, b1, stride_c)
	conv1[conv1<=0] = 0

	conv2 = convolution(conv1, f2, b2, stride_c)
	conv2[conv2<=0] = 0

	pooled = maxpool(conv2, pool, stride_p)

	(nf2, dim2, _) = pooled.shape
	fc = pooled.reshape((nf2 * dim2 * dim2, 1))

	z = w3.dot(fc) + b3
	z[z<=0] = 0

	out = w4.dot(z) + b4

	probs = softmax(out)

	return np.argmax(probs), np.max(probs)

def initializeFilter(size, scale = 1.0):
	stddev = scale / np.sqrt(np.prod(size))
	return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
	return np.random.standard_normal(size=size) * 0.01

def zeroInitialization(params):
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

	f1 = np.zeros(f1.shape)
	f2 = np.zeros(f2.shape)
	w3 = np.zeros(w3.shape)
	w4 = np.zeros(w4.shape)
	b1 = np.zeros(b1.shape)
	b2 = np.zeros(b2.shape)
	b3 = np.zeros(b3.shape)
	b4 = np.zeros(b4.shape)

	mask = [f1, f2, w3, w4, b1, b2, b3, b4]

	return mask