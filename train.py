import matplotlib.pyplot as plt

from utils import *
from network import *

if __name__ == '__main__':
	save_path = 'path.pkl'

	# uncomment the below line to train the network
	# cost = train(save_path=save_path)
	params, cost = pickle.load(open(save_path, 'rb'))
	[f1, f2, w3, w4, b1, b2, b3, b4] = params

	# Plot the cost and iterations
	plt.plot(cost, 'r')
	plt.xlabel("Iterations")
	plt.ylabel('cost')
	plt.legend('loss', loc='upper right')
	# plt.show()

	# extract test data
	m = 10000
	X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
	Y = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m, 1)

	# normalizing the data by subtracting the mean and dividing with standard deviation
	X -= int(np.mean(X))
	X /= int(np.std(X))
	test_data = np.hstack((X, Y))

	X = test_data[:, 0:-1]
	X = X.reshape(len(test_data), 1, 28, 28)
	Y = test_data[:, -1]

	# checking outputs
	# print(Y[:10])

	correct = 0
	digit_count = [0 for i in range(10)]
	digit_correct = [0 for i in range(10)]

	print('Accuracy on test set: ')

	# progrss bar for test data
	t = tqdm(range(len(X)), leave=True)

	# loop over the entire test data
	for i in t:
		x = X[i]
		# predict the probabilities for each image
		pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
		digit_count[int(Y[i])] += 1
		if pred == Y[i]:
			# increment for correct predictions
			correct += 1
			digit_correct[pred] += 1

		t.set_description("Acc: %0.2f" % (float(correct/(i+1))*100))

	# total accuracy
	print("Total Acc: %0.2f" % (float(correct/len(test_data)*100)))
	x = np.arange(10)

	# total recall
	digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
	plt.xlabel('digits')
	plt.ylabel('recall')
	plt.title('Recall on Test data')
	plt.bar(x, digit_recall)
	plt.show()