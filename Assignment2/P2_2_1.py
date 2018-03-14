import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

def MINIST_class()
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

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	dataNum, dataLen = trainData.shape
	target_matrix = np.zero(shape = (dataNum, 10)) # 10 classes
	for i in range(dataNum):
		target_matrix[i, trainTarget[i]] = 1.0
	
	# create placeholder
	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])

	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)

	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias) 
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction)
	entropy_loss =tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = entropy_loss + weight_decay_loss
	max_prob = tf.argmax(prediction,axis=1)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	loss_list, accuracy_list = [], []

	for i in range(140): # iteration = 140
		rand_index = list(range(0, len(trainData)))
		random.shuffle(rand_index)
		for k in range(batch_num):#7
		# 7 batches for each epoch
		# Get the batch
			batch_data, batch_target = [], []
			for j in rand_index[k*batch_size:(k+1)*batch_size]:
				batch_data.append(trainData[rand_index[j]])
				batch_target.append(trainTarget[rand_index[j]])

			# update weight vector for each batch
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
		
		loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: trainTarget}) 	
		loss_list.append(loss_per_epoch)
		
		










