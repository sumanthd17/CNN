import argparse
import matplotlib.pyplot as plt

from util import *
from network import *

if __name == '__main__':
	save_path = 'path.pkl'

	cost = train(save_path=save_path)
	params, cost = pickle.load(open(save_path, 'rb'))
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

	plt.plot(cost, 'r')
	plt.xlabel("Iterations")
	plt.ylabel('cost')
	plt.legend('loss', loc='upper right')
	plt.show()

	x = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
	y = extract_labels('t10k-images-idx3-ubyte.gz', m).reshape(m, 1)

	x -= int(np.mean(x))
	x /= int(np.std(x))
	test_data = np.hstack(x, y)

	X = test_data[:, 0:-1]
	X = x.reshape(len(test_data), 1, 28, 28)
	y = test_data[:, -1]

	correct = 0
	digit_count = [0 for i in range(10)]
	digit_correct = [0 for i in range(10)]

	print('Accuracy on test set: ')

	t = tqdm(range(len(X)), leave=True)

	for i in t:
		x = X[i]
		pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
		digit_count[int(y[i])] += 1
		if pred == y[i]:
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