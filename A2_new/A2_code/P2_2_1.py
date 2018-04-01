import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

def MINIST_class(learning_rate):
	with np.load("notMNIST.npz") as data:
		Data, Target = data ["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]
	
	batch_size = 500
	decay_efficient = 0.01
	batch_num = 30

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	dataNum, dataLen = trainData.shape
	target_matrix = np.zeros(shape = (dataNum, 10)) # 10 classes
	for i in range(dataNum):
		target_matrix[i, trainTarget[i]] = 1.0

	valid_dataNum, valid_dataLen = validData.shape
	valid_target_matrix = np.zeros(shape = (valid_dataNum, 10)) # 10 classes
	for i in range(valid_dataNum):
		valid_target_matrix[i, validTarget[i]] = 1.0

	test_dataNum, test_dataLen = testData.shape
	test_target_matrix = np.zeros(shape = (test_dataNum, 10)) # 10 classes
	for i in range(test_dataNum):
		test_target_matrix[i, testTarget[i]] = 1.0
	
	# create placeholder
	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 10])

	weight = tf.Variable(tf.zeros([10, 28*28]))
	bias = tf.Variable(0.0)

	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias)  # W^T*x+b	
	sig_prediction = tf.sigmoid(tf.add(tf.matmul(data, tf.transpose(weight)), bias))
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction)
	entropy_loss =tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = entropy_loss + weight_decay_loss
	max_prob = tf.argmax(sig_prediction,axis=1)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	train_loss_list, train_accuracy_list = [], []
	valid_loss_list, valid_accuracy_list = [], []

	for i in range(1000): # tune iteration = 140, other = 1000
		rand_index = list(range(0, len(trainData)))
		random.shuffle(rand_index)
		for k in range(batch_num):#7
		# 7 batches for each epoch
		# Get the batch
			batch_data, batch_target = [], []
			for j in rand_index[k*batch_size:(k+1)*batch_size]:
				batch_data.append(trainData[rand_index[j]])
				batch_target.append(target_matrix[rand_index[j]])

			# update weight vector for each batch
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
		
		loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: target_matrix}) 	
		train_loss_list.append(loss_per_epoch)
		
		valid_loss = sess.run(loss, feed_dict={data: validData, target: valid_target_matrix})
		valid_loss_list.append(valid_loss)
		
		train_accuracy_list.append(np.mean(sess.run(max_prob, feed_dict={data: trainData}) == trainTarget))
		valid_accuracy_list.append(np.mean(sess.run(max_prob, feed_dict={data: validData}) == validTarget))

	test_accuracy = np.mean(sess.run(max_prob, feed_dict={data: testData}) == testTarget)
	return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy

def tune():
	#learning_rate = 0.0001
	#step = 0.0002
	l_r = [0.0001,0.0005,0.001,0.005,0.01] # best 0.001
	loss_dicts = {}
	for x in l_r:
		train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy = MINIST_class(x)
		key = str(x)
		loss_dicts[key] = train_loss_list
	
	plt.figure()
	plt.plot(loss_dicts[str(0.0001)])
	plt.plot(loss_dicts[str(0.0005)])
	plt.plot(loss_dicts[str(0.001)])
	plt.plot(loss_dicts[str(0.005)])
	plt.plot(loss_dicts[str(0.01)])
	plt.legend(['0.0001','0.0005','0.001','0.005','0.01'], loc='upper right')
	plt.title("2.2.1 Cross Entropy Loss for different learning rate")
	plt.show()

def plot_bestLR():
	#learning_rate = 0.0001
	#step = 0.0002
	l_r = 0.001
	train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy = MINIST_class(l_r)
	
	print("Best test classification accuracy from the logistic regression model is %f"%(test_accuracy))
	plt.figure()
	plt.plot(train_loss_list)
	plt.plot(valid_loss_list)
	plt.legend(['training loss', 'validation loss'], loc='upper right')
	plt.title("2.2.1 Logistic Regression Cross Entropy Loss for best learning rate 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Cross Entropy Loss")
	plt.grid(True)
	plt.show()

	plt.figure()
	plt.plot(train_accuracy_list)
	plt.plot(valid_accuracy_list)
	plt.legend(['training accuracy', 'validation accuracy'], loc='upper right')
	plt.title("2.2.1 Logistic Regression Accuracy for best learning rate 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True)
	plt.show()
	

plot_bestLR()
#tune()









