import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random


def add_one_layer(input_tensor,num_hidden_units):
	inputSize=input_tensor.get_shape()[1]
	weight = tf.get_variable(name="weights", shape=[inputSize, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
	bias = tf.get_variable(name="biases", shape=[1, num_hidden_units], initializer=tf.constant_initializer(0))
	weighted_sum = tf.add(tf.matmul(input_tensor, weight), bias)
	return weighted_sum;


def loss(learning_rate):
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
	decay_efficient = 0.0003
	epoch = 100
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

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 10])
	
	#create layer
	with tf.variable_scope("hidden_layer1"):
		weight_sum_1 = add_one_layer(data,1000)
		tf.get_variable_scope().reuse_variables()
		weight1=tf.get_variable("weights")

	with tf.variable_scope("output_layer"):
		weight_sum_2 = add_one_layer(tf.nn.relu(weight_sum_1),10)
		tf.get_variable_scope().reuse_variables()
		weight2=tf.get_variable("weights")

	sig_prediction = tf.nn.softmax(logits=weight_sum_2)
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=weight_sum_2)
	entropy_loss =tf.reduce_mean(entropy)
	weight_decay_loss1 = decay_efficient * tf.nn.l2_loss(weight1)
	weight_decay_loss2 = decay_efficient * tf.nn.l2_loss(weight2)
	weight_decay_loss = weight_decay_loss1 + weight_decay_loss2
	loss = entropy_loss + weight_decay_loss
	max_prob = tf.argmax(sig_prediction,axis=1)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	train_loss_list, train_accuracy_list = [], []
	valid_loss_list, valid_accuracy_list = [], []
	test_loss_list, test_accuracy_list = [], []

	for i in range(epoch):  
		rand_index = list(range(0, len(trainData)))
		random.shuffle(rand_index)
		for k in range(batch_num):
			batch_data, batch_target = [], []
			for j in rand_index[k * batch_size:(k + 1) * batch_size]:
			    batch_data.append(trainData[rand_index[j]])
			    batch_target.append(target_matrix[rand_index[j]])

			    # update weight vector for each batch
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})

		loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: target_matrix}) 	
		train_loss_list.append(loss_per_epoch)
		
	
	
	return train_loss_list
	 

def plot():
	train_loss_list1 = loss(0.01)
	tf.reset_default_graph() 
	train_loss_list4 = loss(0.005)
	tf.reset_default_graph() 
	train_loss_list2 = loss(0.001)
	tf.reset_default_graph() 
	train_loss_list3 = loss(0.0001)
	
	plt.figure()
	plt.plot(train_loss_list1)
	plt.plot(train_loss_list4)
	plt.plot(train_loss_list2)
	plt.plot(train_loss_list3)
	plt.legend(['learning rate: 0.01', 'learning rate: 0.005', 'learning rate: 0.001','learning rate: 0.0001'], loc='upper right')
	plt.title("1.1.2 Tuning Learning Rate")
	plt.xlabel("Epoch")
	plt.ylabel("Cross Entropy Loss")
	plt.grid(True)
	plt.show()


plot()
