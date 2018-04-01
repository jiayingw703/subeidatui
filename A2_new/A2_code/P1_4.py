import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time

def SGD():
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
	# best parameters found before
	learning_rate = 0.005
	batch_size = 500
	decay_efficient = 0
	iteration = 20000
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
	#MSE_loss = squared_diff_sum / (2.0*500.0)
	MSE_loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(tf.subtract(prediction, target), 2)))
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	start = time.time()

	for i in range(2857): # 20000 iterations / 7 batches = 2857 epoch
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

	final_training_MSE = sess.run(loss, feed_dict={data: trainData, target: trainTarget})

	end = time.time()
	time_ms = (end - start)*1000.0	
	
	validation_pred = sess.run(prediction, feed_dict={data: validData, target: validTarget})
	val_acc_count = 0.0
	for i in list(range(0, len(validation_pred))):
            if validation_pred[i] > 0.5 and validTarget[i] > 0.5:
                val_acc_count += 1
            elif validation_pred[i] < 0.5 and validTarget[i] < 0.5:
                val_acc_count += 1

	test_pred = sess.run(prediction, feed_dict={data: testData, target: testTarget})
	test_acc_count = 0.0
	for i in list(range(0, len(test_pred))):
            if test_pred[i] > 0.5 and testTarget[i] > 0.5:
                test_acc_count += 1
            elif test_pred[i] < 0.5 and testTarget[i] < 0.5:
                test_acc_count += 1

	return final_training_MSE, val_acc_count/100.0, test_acc_count/145.0, time_ms

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
	trainTarget = trainTarget.astype(np.float32)
	X = tf.Variable(trainData)
	Y = tf.Variable(trainTarget)

	
	start = time.time()
	weight = tf.matrix_solve_ls(X, Y, l2_regularizer=0.0, fast=True)
	end = time.time()
	time_ms = (end - start)*1000.0
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())	
	prediction = tf.matmul(trainData, weight)
	#squared_diff_sum = tf.reduce_sum(tf.pow((prediction - Y), 2))
	#training_MSE_loss = squared_diff_sum / (2.0*500.0)
	training_MSE_loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(tf.subtract(prediction, Y), 2)))
	validation_pred = tf.matmul(validData, weight)
	val_acc_count = 0.0
	

	for i in list(range(0, len(validTarget))):
            if sess.run(tf.greater(validation_pred[i],0.5)) and validTarget[i] > 0.5:
                val_acc_count += 1
            elif sess.run(tf.less(validation_pred[i],0.5)) and validTarget[i] < 0.5:
                val_acc_count += 1

	test_pred = tf.matmul(testData, weight)
	test_acc_count = 0.0
	for i in list(range(0, len(testTarget))):
            if sess.run(tf.greater(test_pred[i],0.5)) and testTarget[i] > 0.5:
                test_acc_count += 1
            elif sess.run(tf.less(test_pred[i],0.5)) and testTarget[i] < 0.5:
                test_acc_count += 1

	return training_MSE_loss, val_acc_count/100.0, test_acc_count/145.0, time_ms
	
def plot():
	training_MSE, val_accuracy, test_accuracy, time_ms = SGD()
	#print training_MSE
	print("SGD result:\n")
	print("training_MSE: %f, val_accuracy: %f, test_accuracy: %f, time_ms: %f\n"%(training_MSE, val_accuracy, test_accuracy, time_ms))

	#training_MSE, val_accuracy, test_accuracy, time_ms = normal()

	#sess = tf.Session()
	#sess.run(tf.global_variables_initializer())
	#print("Normal equation result:\n")	
	#print("validation accuracy: %f,test accuracy:%f ,time_ms:%f\n" %(val_accuracy, test_accuracy, time_ms))
	#print sess.run(training_MSE)
     

	#print("training_MSE: %f, val_accuracy: %f, test_accuracy: %f, time_ms: %f\n"%(training_MSE, val_accuracy, test_accuracy, time_ms))

plot()
