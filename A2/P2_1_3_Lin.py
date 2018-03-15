import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy



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

	training_loss,valid_loss = [],[]
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
		loss_per_epoch = sess.run(loss, feed_dict={data: validData, target: validTarget})
		valid_loss.append(loss_per_epoch)
	return training_loss,valid_loss

	
def plot():
	#train_accy_list, train_loss_list = Logistic_Reg(0.001)
	MSE,valid_loss=lin_Reg(0.001)
	plt.figure()
	#plt.plot(trainData,trainTarget,'.')
	#plt.plot(validtion)#,'k', label = "Learning Rate: 0.005")
	plt.plot(valid_loss)
	plt.plot(MSE)#, 'k--', label = "Learning Rate: 0.001")

	#plt.plot(training_loss_3)#, 'k-', label = "Learning Rate: 0.0001")
	#legend = ax.legend(loc="upper center", shadow=False)
	#frame = legend.get_frame()
	#frame.set_facecolor('0.90')
	plt.legend(['Validation Loss','Training Loss'], loc='upper right')
	plt.title("2.1.3 MSE for Linear Regression")
	plt.ylabel('MSE')
	plt.xlabel('Epoch')
	plt.grid(True)
	#plt.title("Learning rate: %f"%learning_rate)
	plt.show()
plot()
