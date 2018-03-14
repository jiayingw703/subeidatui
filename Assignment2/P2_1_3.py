import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

#2.1 logistic

def Logistic_Reg(learning_rate):
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
	decay_efficient = 0
	iteration = 5000
	batch_num = 7

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])
	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)
	#learning_rate = tf.placeholder("float32", name = "learning_rate")
	#loss fn
	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias) #W^T*x+b
	#squared_diff_sum = tf.reduce_sum(tf.pow((prediction - target) , 2))
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prediction)
	MSE_loss =tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(714): # 20000 iterations / 7 batches = 2857 epoch
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

	train_pred = sess.run(prediction, feed_dict={data: trainData, target: trainTarget})
	train_acc_count = 0.0
	for i in list(range(0, len(trainTarget))):
            if sess.run(tf.greater(train_pred[i],0.5)) and trainTarget[i] > 0.5:
                train_acc_count += 1
            elif sess.run(tf.less(train_pred[i],0.5)) and trainTarget[i] < 0.5:
                train_acc_count += 1

	validation_pred = sess.run(prediction, feed_dict={data: validData, target: validTarget})
	val_acc_count = 0.0	
	for i in list(range(0, len(validTarget))):
            if sess.run(tf.greater(validation_pred[i],0.5)) and validTarget[i] > 0.5:
                val_acc_count += 1
            elif sess.run(tf.less(validation_pred[i],0.5)) and validTarget[i] < 0.5:
                val_acc_count += 1

	test_pred = sess.run(prediction, feed_dict={data: testData, target: testTarget})
	test_acc_count = 0.0
	for i in list(range(0, len(testTarget))):
            if sess.run(tf.greater(test_pred[i],0.5)) and testTarget[i] > 0.5:
                test_acc_count += 1
            elif sess.run(tf.less(test_pred[i],0.5)) and testTarget[i] < 0.5:
                test_acc_count += 1

	return train_acc_count/3500.0, val_acc_count/100.0, test_acc_count/145.0

def lin_Reg(learning_rate):
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
	decay_efficient = 0
	iteration = 5000
	batch_num = 7

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])
	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)
	#learning_rate = tf.placeholder("float32", name = "learning_rate")
	#loss fn
	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias) #W^T*x+b
	#squared_diff_sum = tf.reduce_sum(tf.pow((prediction - target) , 2))
	#MSE_loss = tf.scalar_mul(0.001,squared_diff_sum )
	MSE_loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(tf.subtract(prediction, target), 2)))
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	training_loss = []
	for i in range(714): # 20000 iterations / 7 batches = 2857 epoch
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
		training_loss.append(loss_per_epoch)
	return training_loss


def normal():
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

	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)
	trainTarget_cp = copy.deepcopy(trainTarget)
	trainTarget = trainTarget.astype(np.float32)
	X = tf.Variable(trainData)
	Y = tf.Variable(trainTarget)

	weight = tf.matrix_solve_ls(X, Y, l2_regularizer=0.0, fast=True)

	prediction = tf.matmul(trainData, weight)
	#squared_diff_sum = tf.reduce_sum(tf.pow((prediction - Y), 2))
	#training_MSE_loss = squared_diff_sum / (2.0*500.0)

	
	validation_pred = tf.matmul(validData, weight)
	val_acc_count = 0.0
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())	

#	for i in list(range(0, len(validTarget))):
#            if sess.run(tf.greater(validation_pred[i],0.5)) and validTarget[i] > 0.5:
#                val_acc_count += 1
#            elif sess.run(tf.less(validation_pred[i],0.5)) and validTarget[i] < 0.5:
#                val_acc_count += 1

#	test_pred = tf.matmul(testData, weight)
	test_acc_count = 0.0
#	for i in list(range(0, len(testTarget))):
#            if sess.run(tf.greater(test_pred[i],0.5)) and testTarget[i] > 0.5:
#                test_acc_count += 1
#            elif sess.run(tf.less(test_pred[i],0.5)) and testTarget[i] < 0.5:
#                test_acc_count += 1

	train_pred = tf.matmul(trainData, weight)
	train_acc_count = 0.0
	for i in list(range(0, len(trainTarget))):
            if sess.run(tf.greater(train_pred[i],0.5)) and trainTarget_cp[i] > 0.5:
                train_acc_count += 1
            elif sess.run(tf.less(train_pred[i],0.5)) and trainTarget_cp[i] < 0.5:
                train_acc_count += 1

	return train_acc_count/3500.0, val_acc_count/100.0, test_acc_count/145.0


	
def plot():
	#log_train_acc, log_valid_acc, log_test_acc= Logistic_Reg(0.001)
	normal_train_acc, normal_valid_acc, normal_test_acc = normal()
	
	#print ("Logistic Regression: train accuracy: %f, validation accuracy: %f, test accuracy: %f"%(log_train_acc, log_valid_acc, log_test_acc))
	print ("Normal equation: train accuracy: %f, validation accuracy: %f, test accuracy: %f"%(normal_train_acc, normal_valid_acc, normal_test_acc))
plot()
