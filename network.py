from ops import *
from backward import *
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

	z = we.dot(fc) + b3
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

	dfc = w3.dot(dz)
	dpool = dfc.reshape(pooles.shape)

	dconv2 = maxpoolBackward(dpool, conv2, pool, stride_p)
	dconv2[conv2<=0] = 0

	dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, stride_c)
	dconv1[conv1<=0] = 0

	dimage, df1, db1 = convolutionBackward(dconv1, image, f1, stride_c)

	grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

	return grads, loss

def train(num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, batch_size=32, num_epochs=2, save_path='params.pkl'):
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
    	batches = [train_data[k: k + batch_size] for k in range(0, train_data.shappe[0], batch_size)]

    	t = tqdm(batches)
    	for x, batch in enumerate(t):
    		params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
    		t.set_description("Cost: %.2f" % (cost[-1]))

	to_save = [params, cost]

	with open(save_path, 'wb') as file:
		pickle.dump(to_save, file)

	return cost