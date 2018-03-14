import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

#2_1 for classification


	# parameters
def loss(learning_rate):
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
	decay_efficient = 0.01
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
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	train, valid = [],[]
	for i in range(714): # 5000 iterations / 7 batches = 714 epoch
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

		#loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: trainTarget})
		
	validation_pred = sess.run(prediction, feed_dict={data: validData, target: validTarget})
	acc_count = 0.0
	for i in list(range(0, len(validation_pred))):
		     if validation_pred[i] > 0.5 and validTarget[i] > 0.5:
		         acc_count += 1
		     elif validation_pred[i] < 0.5 and validTarget[i] < 0.5:
		          acc_count += 1
	
            	
	return acc_count/100.0


		

	
def plot():
	#learning_rate = 0.0001
	#step = 0.0002
	l_r = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
	dicts = {}
	for x in l_r:
		loss_v = loss(x)
		key = str(x)
		dicts[key] = loss_v;
	#plt.figure()
	print ('Tuning learning rate by validation data for classification.')
	for y in dicts:
		print ('learning rate:',y,' Validation Final classification:', dicts[y])
	print ('learning rate for min validation accuracy classification:' ,max(dicts,key=dicts.get)) 
		#plt.plot(dicts[y])
		
	#plt.legend(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], loc='upper right')
		
		
	#plt.show()
	
	#fig, ax = plt.plots()
	

	#fig, ax = plt.plots()
	#plt.figure()
	#plt.plot(trainData,trainTarget,'.')
	#plt.plot(training_loss_1)#,'k', label = "Learning Rate: 0.005")
	#plt.plot(training_loss_2)#, 'k--', label = "Learning Rate: 0.001")
	#plt.plot(training_loss_3)#, 'k-', label = "Learning Rate: 0.0001")
	#legend = ax.legend(loc="upper center", shadow=False)
	#frame = legend.get_frame()
	#frame.set_facecolor('0.90')
	#plt.legend(['Learning Rate: 0.005','Learning Rate: 0.001','Learning Rate: 0.0001'], loc='upper right')
	#plt.title("1.1 Learning Rate")
	#plt.title("Learning rate: %f"%learning_rate)
	#plt.show()
plot()
