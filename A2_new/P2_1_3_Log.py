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

	prediction = tf.add(tf.matmul(data, tf.transpose(weight)), bias) #W^T*x+b
	sig_prediction = tf.sigmoid(tf.add(tf.matmul(data, tf.transpose(weight)), bias))
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prediction)
	MSE_loss =tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = MSE_loss + weight_decay_loss
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	train_accy_list, valid_accy_list= [], []
	training_loss,valid_loss = [],[]

	for i in range(714): # 5000 iterations / 7 batches = 2857 epoch
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
		# store loss
		loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: trainTarget})
		training_loss.append(loss_per_epoch)
		loss_per_epoch = sess.run(loss, feed_dict={data: validData, target: validTarget})
		valid_loss.append(loss_per_epoch)
		# store accuracy		
		train_pred = sess.run(sig_prediction, feed_dict={data: trainData, target: trainTarget})
		train_acc_count = 0.0
		for i in list(range(0, len(trainTarget))):
		    if train_pred[i] > 0.5 and trainTarget[i] > 0.5:
		        train_acc_count += 1
		    elif train_pred[i] < 0.5 and trainTarget[i] < 0.5:
		        train_acc_count += 1
		train_accy_list.append(train_acc_count/3500.0)

		validation_pred = sess.run(sig_prediction, feed_dict={data: validData, target: validTarget})
		val_acc_count = 0.0	
		for i in list(range(0, len(validTarget))):
		    if validation_pred[i]>0.5 and validTarget[i] > 0.5:
		        val_acc_count += 1
		    elif validation_pred[i]<0.5 and validTarget[i] < 0.5:
		        val_acc_count += 1
		valid_accy_list.append(val_acc_count/100.0)

	return training_loss, valid_loss, train_accy_list, valid_accy_list


def plot():
	training_loss, valid_loss, train_accy_list, valid_accy_list = Logistic_Reg(0.001)

	plt.figure()
	plt.plot(training_loss)
	plt.plot(valid_loss)
	plt.legend(['training loss', 'validation loss'], loc='upper right')
	plt.title("2.1.3 Logistic Regression cross entropy for learning rate 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Cross Entropy Loss")
	plt.grid(True)
	plt.show()

	plt.figure()
	plt.plot(train_accy_list)
	plt.plot(valid_accy_list)
	plt.legend(['training accuracy', 'validation accuracy'], loc='upper right')
	plt.title("2.1.3 Logistic Regression Accuracy for learning rate 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True)
	plt.show()
plot()
