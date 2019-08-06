import argparse
import matplotlib.pyplot as plt

from utils import *
from network import *

if __name__ == '__main__':
	save_path = 'path.pkl'

	# cost = train(save_path=save_path)
	params, cost = pickle.load(open(save_path, 'rb'))
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

	plt.plot(cost, 'r')
	plt.xlabel("Iterations")
	plt.ylabel('cost')
	plt.legend('loss', loc='upper right')
	# plt.show()

	m = 10000
	X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
	Y = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m, 1)

	X -= int(np.mean(X))
	X /= int(np.std(X))
	test_data = np.hstack((X, Y))

	X = test_data[:, 0:-1]
	X = X.reshape(len(test_data), 1, 28, 28)
	Y = test_data[:, -1]

	print(Y[:10])

	correct = 0
	digit_count = [0 for i in range(10)]
	digit_correct = [0 for i in range(10)]

	print('Accuracy on test set: ')

	t = tqdm(range(len(X)), leave=True)

	for i in t:
		x = X[i]
		pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
		digit_count[int(Y[i])] += 1
		if pred == Y[i]:
			correct += 1
			digit_correct[pred] += 1

		t.set_description("Acc: %0.2f" % (float(correct/(i+1))*100))

	print("Total Acc: %0.2f" % (float(correct/len(test_data)*100)))
	x = np.arrange(10)
	digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
	plt.xlabel('digits')
	plt.ylabel('recall')
	plt.title('Recall on Test data')
	plt.bar(x, digit_recall)
	plt.show()