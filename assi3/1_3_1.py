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



def loss_drop_out():
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
	
	learning_rate = 0.001
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
		W_X_1 = add_one_layer(data,1000)
		W_X_1_dropout = tf.nn.dropout(W_X_1, 0.5)
		tf.get_variable_scope().reuse_variables()
		weight1=tf.get_variable("weights")

	with tf.variable_scope("output_layer"):
		W_X_2_dropout = add_one_layer(tf.nn.relu(W_X_1_dropout),10)
		tf.get_variable_scope().reuse_variables()
		weight2=tf.get_variable("weights")
		bias2 = tf.get_variable("biases")
		y_pred = tf.add(tf.matmul(W_X_1, weight2), bias2)#for prediction
		

	sig_prediction = tf.nn.softmax(logits=y_pred)
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=W_X_2_dropout)
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

		train_accuracy_list.append(1.0-np.mean(sess.run(max_prob, feed_dict={data: trainData}) == trainTarget))
		valid_accuracy_list.append(1.0-np.mean(sess.run(max_prob, feed_dict={data: validData}) == validTarget))
		test_accuracy_list.append(1.0-np.mean(sess.run(max_prob, feed_dict={data: testData}) == testTarget))

	return train_accuracy_list, valid_accuracy_list,test_accuracy_list


def plot():
	train_accuracy_list,valid_accuracy_list,test_accuracy_list= loss_drop_out()
	

	plt.figure()
	plt.plot(train_accuracy_list)
	plt.plot(valid_accuracy_list)
	plt.plot(test_accuracy_list)
	plt.legend(['training error', 'validation error', 'test error'], loc='upper right')
	#plt.legend(['training accuracy'], loc='upper right')
	plt.title("1.3.1 Classification error with dropout rate 0.5")
	print("1.3.1 with dropout. Final trainning error :%f, Final validation error:%f, Final test error :%f."%(train_accuracy_list[-1],valid_accuracy_list[-1],test_accuracy_list[-1]))
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	plt.grid(True)
	plt.show()

	

plot()


