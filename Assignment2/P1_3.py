import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

def loss(decay_efficient):
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
	iteration = 20000
	batch_num = 7
	batch_size = 500

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
	validData = validData.reshape((-1, 28*28)).astype(np.float32)
	testData = testData.reshape((-1, 28*28)).astype(np.float32)

	data = tf.placeholder(tf.float32, [None, 28*28])
	target = tf.placeholder(tf.float32, [None, 1])
	weight = tf.Variable(tf.zeros([1, 28*28]))
	bias = tf.Variable(0.0)
	#learning_rate = tf.placeholder("float32", name = "learning_rate")

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
	
	#final_training_MSE = sess.run(loss, feed_dict={data: trainData, target: trainTarget})
	validation_pred = sess.run(prediction, feed_dict={data: validData, target: validTarget})
	acc_count = 0.0
	for i in list(range(0, len(validation_pred))):
            if validation_pred[i] > 0.5 and validTarget[i] > 0.5:
                acc_count += 1
            elif validation_pred[i] < 0.5 and validTarget[i] < 0.5:
                acc_count += 1
	final_training_MSE = sess.run(MSE_loss, feed_dict={data: validData, target: validTarget})      	
	return acc_count/10000.0,final_training_MSE
	
def plot():
	decay_efficient = 0.0
	validation_accuracy_1,MSE1 = loss(decay_efficient)

	decay_efficient = 0.001
	validation_accuracy_2,MSE2 = loss(decay_efficient)

	decay_efficient = 0.1
	validation_accuracy_3,MSE3 = loss(decay_efficient)

	decay_efficient = 1.0
	validation_accuracy_4,MSE4 = loss(decay_efficient)

	print ("validation accuracy for decay_efficient 0.0: %f, MSE: %f"%(validation_accuracy_1,MSE1))
	print ("validation accuracy for decay_efficient 0.001: %f, MSE: %f"%(validation_accuracy_2,MSE2))
	print ("validation accuracy for decay_efficient 0.1: %f, MSE: %f"%(validation_accuracy_3,MSE3))
	print ("validation accuracy for decay_efficient 1.0: %f, MSE: %f"%(validation_accuracy_4,MSE4))
plot()
