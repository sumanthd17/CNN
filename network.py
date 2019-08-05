from foward import *
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