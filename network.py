from ops import *
from utils import *

import numpy as np
import pickle
from tqdm import tqdm

def conv(image, label, params, stride_c = 1, pool = 2, stride_p = 2):
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

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

	loss = categoricalCrossEntropy(probs, label)

	# Backward propagation
	dout = probs - label
	dw4 = dout.dot(z.T)
	db4 = np.sum(dout, axis = 1).reshape(b4.shape)

	dz = w4.T.dot(dout)
	dz[z<=0] = 0
	dw3 = dz.dot(fc.T)
	db3 = np.sum(dz, axis = 1).reshape(b3.shape)

	dfc = w3.T.dot(dz)
	dpool = dfc.reshape(pooled.shape)

	dconv2 = maxpoolBackward(dpool, conv2, pool, stride_p)
	dconv2[conv2<=0] = 0

	dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, stride_c)
	dconv1[conv1<=0] = 0

	dimage, df1, db1 = convolutionBackward(dconv1, image, f1, stride_c)

	grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

	return grads, loss

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

	X = batch[:, 0:-1]
	X = X.reshape(len(batch), n_c, dim, dim)
	Y = batch[:, -1]

	cost_ = 0
	batch_size = len(batch)

	[df1, df2, dw3, dw4, db1, db2, db3, db4] = zeroInitialization(params)
	[v1, v2, v3, v4, bv1, bv2, bv3, bv4] = zeroInitialization(params)
	[s1, s2, s3, s4, bs1, bs2, bs3, bs4] = zeroInitialization(params)

	for i in range(batch_size):
		x = X[i]
		y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)

		grads, loss = conv(x, y, params, 1, 2, 2)
		[df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

		df1 += df1_
		db1 += db1_
		df2 += df2_
		db2 += db2_
		dw3 += dw3_
		db3 += db3_
		dw4 += dw4_
		db4 += db4_

		cost_ += loss
		
	v1 = beta1 * v1 + (1- beta1) * df1 / batch_size
	s1 = beta2 * s1 + (1 - beta2) * (df1/batch_size)**2
	f1 -= lr * v1 / np.sqrt(s1 + 1e-7)

	bv1 = beta1 * bv1 + (1- beta1) * db1 / batch_size
	bs1 = beta2 * bs1 + (1 - beta2) * (db1/batch_size)**2
	b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

	v2 = beta1 * v2 + (1- beta1) * df2 / batch_size
	s2 = beta2 * s2 + (1 - beta2) * (df2/batch_size)**2
	f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

	bv2 = beta1 * bv2 + (1- beta1) * db2 / batch_size
	bs2 = beta2 * bs2 + (1 - beta2) * (db2/batch_size)**2
	b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

	v3 = beta1 * v3 + (1- beta1) * dw3 / batch_size
	s3 = beta2 * s3 + (1 - beta2) * (dw3/batch_size)**2
	w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

	bv3 = beta1 * bv3 + (1- beta1) * db3 / batch_size
	bs3 = beta2 * bs3 + (1 - beta2) * (db3/batch_size)**2
	b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

	v4 = beta1 * v4 + (1- beta1) * dw4 / batch_size
	s4 = beta2 * s4 + (1 - beta2) * (dw4/batch_size)**2
	w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

	bv4 = beta1 * bv4 + (1- beta1) * db4 / batch_size
	bs4 = beta2 * bs4 + (1 - beta2) * (db4/batch_size)**2
	b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

	cost_ = cost_ / batch_size
	cost.append(cost_)

	params = [f1, f2, w3, w4, b1, b2, b3, b4]

	return params, cost



def train(num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8, batch_size=32, num_epochs=2, save_path='params.pkl'):
	m = 50000
	X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
	y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
	X -= int(np.mean(X))
	X /= int(np.std(X))
	train_data = np.hstack((X,y_dash))

	np.random.shuffle(train_data)

	f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 800), (10, 128)
	f1 = initializeFilter(f1)
	f2 = initializeFilter(f2)
	w3 = initializeWeight(w3)
	w4 = initializeWeight(w4)

	b1 = np.zeros((f1.shape[0],1))
	b2 = np.zeros((f2.shape[0],1))
	b3 = np.zeros((w3.shape[0],1))
	b4 = np.zeros((w4.shape[0],1))

	params = [f1, f2, w3, w4, b1, b2, b3, b4]

	cost = []
	print("Learning rate: " + str(lr) + " Batch size: " + str(batch_size))

	for epoch in range(num_epochs):
		np.random.shuffle(train_data)
		batches = [train_data[k: k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

		t = tqdm(batches)
		for x, batch in enumerate(t):
			params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
			t.set_description("Cost: %.2f" % (cost[-1]))

	to_save = [params, cost]

	with open(save_path, 'wb') as file:
		pickle.dump(to_save, file)

	return cost