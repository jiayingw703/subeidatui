import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import time

def loss(batch_size):
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
	learning_rate = 0.005
	decay_efficient = 0
	iteration = 20000

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])
	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)
	#batch_size_var=tf.placeholder("float32")
	#learning_rate = tf.placeholder("float32", name = "learning_rate")
	#loss fn
	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias) #W^T*x+b
	#squared_diff_sum = tf.reduce_sum(tf.pow((prediction - target) , 2))
	#MSE_loss = squared_diff_sum / (2.0*batch_size_var)
	MSE_loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(tf.subtract(prediction, target), 2)))
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	start = time.time()
	if (3500%batch_size == 0):
		epoch = int(math.floor(iteration/(3500.0/batch_size)))
		print epoch
		for i in range(epoch):
			rand_index = list(range(0, len(trainData)))
			random.shuffle(rand_index)
			batch_num = int(3500/batch_size)
			for k in range(batch_num):
			# Get the batch
				batch_data, batch_target = [], []
				for j in rand_index[k*batch_size:(k+1)*batch_size]:
					batch_data.append(trainData[rand_index[j]])
					batch_target.append(trainTarget[rand_index[j]])

				# update weight vector for each batch
				batch_data = np.array(batch_data)
				batch_target = np.array(batch_target)
				sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
	else:
		epoch = int(math.floor(iteration/(math.ceil(3500.0/batch_size))))
		print epoch
		for i in range(epoch):
			rand_index = list(range(0, len(trainData)))
			random.shuffle(rand_index)
			batch_num = 2
			for k in range(batch_num):
			# Get the full batches
				batch_data, batch_target = [], []
				for j in rand_index[k*batch_size:(k+1)*batch_size]:
					batch_data.append(trainData[rand_index[j]])
					batch_target.append(trainTarget[rand_index[j]])

				# update weight vector for each batch
				batch_data = np.array(batch_data)
				batch_target = np.array(batch_target)
				sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
			# last incomplete batch
			batch_data, batch_target = [], []
			for j in rand_index[batch_num*batch_size:]:
				batch_data.append(trainData[rand_index[j]])
				batch_target.append(trainTarget[rand_index[j]])
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})

	end = time.time()
	time_ms = (end - start)*1000.0
	final_training_MSE = sess.run(loss, feed_dict={data: trainData, target: trainTarget})
	return final_training_MSE, time_ms
	
def plot():
	batch_size = 500
	training_loss_1, training_time_1 = loss(batch_size)

	batch_size = 1500
	training_loss_2, training_time_2 = loss(batch_size)

	batch_size = 3500
	training_loss_3, training_time_3 = loss(batch_size)

	print ("training loss for batch size 500 %f, training time %f ms"%(training_loss_1, training_time_1))
	print ("training loss for batch size 1500 %f, training time %f ms"%(training_loss_2, training_time_2))
	print ("training loss for batch size 3500 %f, training time %f ms"%(training_loss_3, training_time_3))
plot()
