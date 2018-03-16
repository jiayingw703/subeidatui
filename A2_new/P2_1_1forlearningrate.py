import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

#2_1 for classification


def loss(learning_rate):
	with np.load("notMNIST.npz") as data :
		Data, Target = data ["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
	batch_size = 500
	decay_efficient = 0.01
	iteration = 5000
	batch_num = 7

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])
	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)
	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias)  # W^T*x+b
	# squared_diff_sum = tf.reduce_sum(tf.pow((prediction - target) , 2))
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prediction)
	MSE_loss = tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	training_loss = []
	for i in range(714):  # 20000 iterations / 7 batches = 2857 epoch
		rand_index = list(range(0, len(trainData)))
		random.shuffle(rand_index)
		for k in range(batch_num):  # 7
		# 7 batches for each epoch
		# Get the batch
			batch_data, batch_target = [], []
			for j in rand_index[k * batch_size:(k + 1) * batch_size]:
			    batch_data.append(trainData[rand_index[j]])
			    batch_target.append(trainTarget[rand_index[j]])

			    # update weight vector for each batch
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})

			loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: trainTarget})
			training_loss.append(loss_per_epoch)
	return training_loss

def plot():
	l_r = [0.0001, 0.001, 0.005]
	dicts = {}
	for x in l_r:
		loss_v = loss(x)
		key = str(x)
		dicts[key] = loss_v
	#plt.figure()
	
    	plt.plot(dicts[str(0.0001)])
    	#plt.plot(dicts[str(0.0005)])
      #for extra state in the report we know we have better option as stated in 1.1, however, to keep consistent, we choose among the ones we got provided
    	plt.plot(dicts[str(0.001)])
	plt.plot(dicts[str(0.005)])
	
	plt.legend(['0.0001','0.001','0.005'], loc='upper right')
	plt.title("2.1.1 Tuning Learning Rate Using Cross Entropy Loss")
	plt.xlabel('Cross Entropy Loss')
	plt.ylabel('Epoch')
	plt.grid(True)
	plt.show()
plot()
