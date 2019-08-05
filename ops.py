import numpy as np

def categoricalCrossEntropy(probs, label):
	return -np.sum(label * np.log(probs))

def convolution(image, mask, bias, stride):
	(n_f, n_c_f, f, _) = mask.shape
	n_c, in_dim, _ = image.shape

	out_dim = int((in_dim - f) / stride) + 1

	assert m_c == n_c_f, "Number of channels should match"

	out = np.zeros((n_f, out_dim, out_dim))

	for curr_f in range(n_f):
		curr_y = out_y = 0
		while curr_y + f <= in_dim:
			curr_x = out_x = 0
			while curr_x + f <= in_dim:
				out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
				curr_x += stride
				out_x += 1
			curr_y += stride
			out_y += 1

	return out

def maxpool(image, mask, stride):
	n_c, h, w = imahe.shape

	out = int((h - mask)/stride) + 1

	dowsampled = np.zeros((n_c, out, out))

	for i in range(n_c):
		curr_y = out_y = 0
		while curr_y + mask <= h:
			curr_x = out_x = 0
			while curr_x + mask <= w:
				dowsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+mask, curr_x:curr_x+mask])
				curr_x += stride
				out_x += 1
			curr_y += stride
			out_y += 1

	return dowsampled

def softmax(x):
	out = np.exp(x)
	return out / np.sum(out)

def convolutionBackward(dconv_prev, conv_in, filt, s):
	(n_f, n_c, f, _) = filt.shape
	(_, org_dim, _) = conv_in.shape

	dout = np.zeros(conv_in.shape)
	dfilt = np.zeros(filt.shape)
	dbias = np.zeros((n_f, 1))

	for curr_f in range(n_f):
		curr_y = out_y = 0
		while curr_y + f <= org_dim:
			curr_x = out_x = 0
			while curr_x + f <= org_dim:
				dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x: curr_x+f]
				dout[:, curr_y: curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
				curr_x += s
				out_x += 1
			curr_y += s
			out_y += 1

		dbias[curr_f] = np.sum(dconv_prev[curr_f])

	return dout, dfilt, dbias

def maxpoolBackward(dpool, org, f, s):
	(n_c, org_dim, _) = org.shape

	dout = np.zeros(org.shape)

	for curr_c in range(n_c):
		curr_y = out_y = 0
		while curr_y +f <= org_dim:
			curr_x = out_x = 0
			while curr_x + f <= org_dim:
				(a, b) = nanargmax(org[curr_c, curr_y: curr_y+f, curr_x: curr_x+f])
				dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]

				curr_x += s
				out_x += 1
			curr_y += s
			out_y += 1

		return dout
