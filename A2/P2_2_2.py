import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255.0
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def faceScrub_class(decay_efficient, learning_rate, trainData, validData, testData, trainTarget, validTarget, testTarget):	
	batch_size = 300

	# reshape 3d array to 2d array 
	trainData = trainData.reshape((-1, 32*32)).astype(np.float32)
	validData = validData.reshape((-1, 32*32)).astype(np.float32)
	testData = testData.reshape((-1, 32*32)).astype(np.float32)

	dataNum, dataLen = trainData.shape
	target_matrix = np.zeros(shape = (dataNum, 6)) # 6 classes
	for i in range(dataNum):
		target_matrix[i, trainTarget[i]] = 1.0

	valid_dataNum, valid_dataLen = validData.shape
	valid_target_matrix = np.zeros(shape = (valid_dataNum, 6)) # 6 classes
	for i in range(valid_dataNum):
		valid_target_matrix[i, validTarget[i]] = 1.0

	test_dataNum, test_dataLen = testData.shape
	test_target_matrix = np.zeros(shape = (test_dataNum, 6)) # 6 classes
	for i in range(test_dataNum):
		test_target_matrix[i, testTarget[i]] = 1.0
	
	# create placeholder
	data = tf.placeholder(tf.float32, [None, 32*32])
	target = tf.placeholder(tf.float32, [None, 6])

	weight = tf.Variable(tf.zeros([6, 32*32]))
	bias = tf.Variable(0.0)

	prediction = tf.sigmoid(tf.add(tf.matmul(data, tf.transpose(weight)), bias)) 
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction)
	entropy_loss =tf.reduce_mean(entropy)
	weight_decay_loss = decay_efficient * tf.nn.l2_loss(weight)
	loss = entropy_loss + weight_decay_loss
	max_prob = tf.argmax(prediction,axis=1)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	train_loss_list, train_accuracy_list = [], []
	valid_loss_list, valid_accuracy_list = [], []

	for i in range(1000): # tune iteration = 500, other 1000
		rand_index = list(range(0, len(trainData)))
		random.shuffle(rand_index)
		batch_num = 2 # total training data number 747
		for k in range(batch_num):
		# Get the batch
			batch_data, batch_target = [], []
			for j in rand_index[k*batch_size:(k+1)*batch_size]:
				batch_data.append(trainData[rand_index[j]])
				batch_target.append(target_matrix[rand_index[j]])

			# update weight vector for each batch
			batch_data = np.array(batch_data)
			batch_target = np.array(batch_target)
			sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
		# last incomplete batch
		batch_data, batch_target = [], []
		for j in rand_index[batch_num*batch_size:]:
			batch_data.append(trainData[rand_index[j]])
			batch_target.append(target_matrix[rand_index[j]])
		batch_data = np.array(batch_data)
		batch_target = np.array(batch_target)	
		sess.run(optimizer, feed_dict={data: batch_data, target: batch_target})
		
		loss_per_epoch = sess.run(loss, feed_dict={data: trainData, target: target_matrix}) 	
		train_loss_list.append(loss_per_epoch)
		
		valid_loss = sess.run(loss, feed_dict={data: validData, target: valid_target_matrix})
		valid_loss_list.append(valid_loss)
		
		train_accuracy_list.append(np.mean(sess.run(max_prob, feed_dict={data: trainData}) == trainTarget))
		valid_accuracy_list.append(np.mean(sess.run(max_prob, feed_dict={data: validData}) == validTarget))

	test_accuracy = np.mean(sess.run(max_prob, feed_dict={data: testData}) == testTarget)
	return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy

def tune():
	trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0) # task 0: name

	l_r = [0.0001,0.0005,0.001] # best 0.0001
	decay_efficient = [0.0, 0.001, 0.1, 1] # best 0.001
	loss_dicts = {}
	for x in l_r:
		for y in decay_efficient:
			train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy = faceScrub_class(x,y, trainData, validData, testData, trainTarget, validTarget, testTarget)
			key = str(x)+str(y)
			loss_dicts[key] = train_loss_list
	
	plt.figure()
	#plt.plot(loss_dicts[str(0.0001)+ str(0.0)])
	#plt.plot(loss_dicts[str(0.0005)+ str(0.0)])
	#plt.plot(loss_dicts[str(0.001)+ str(0.0)])
	plt.plot(loss_dicts[str(0.0001)+ str(0.001)])
	plt.plot(loss_dicts[str(0.0005)+ str(0.001)])
	plt.plot(loss_dicts[str(0.001)+ str(0.001)])
	#plt.plot(loss_dicts[str(0.0001)+ str(0.1)])
	#plt.plot(loss_dicts[str(0.0005)+ str(0.1)])
	#plt.plot(loss_dicts[str(0.001)+ str(0.1)])
	#plt.plot(loss_dicts[str(0.0001)+ str(1)])
	#plt.plot(loss_dicts[str(0.0005)+ str(1)])
	#plt.plot(loss_dicts[str(0.001)+ str(1)])
	#plt.legend(['0.0001+0.0','0.0005+0.0','0.01+0.0','0.0001+0.001','0.0005+0.001','0.01+0.001','0.0001+0.01','0.0005+0.01','0.01+0.01','0.0001+1','0.0005+1','0.01+1'], loc='upper right')
	plt.legend(['0.0001+0.001','0.0005+0.001','0.01+0.001'],loc='upper right')
	plt.title("2.2.2 Cross Entropy Loss for different learning rate and decay_efficient")
	plt.show()

def plot_bestLR():
	trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 0) # task 0: name

	train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_accuracy = faceScrub_class(0.0001, 0.001, trainData, validData, testData, trainTarget, validTarget, testTarget)
	
	print("Best test classification accuracy from the logistic regression model is %f"%(test_accuracy))
	plt.figure()
	plt.plot(train_loss_list)
	plt.plot(valid_loss_list)
	plt.legend(['training loss', 'validation loss'], loc='upper right')
	plt.title("2.2.2 Cross Entropy Loss for best learning rate 0.0001 and weight decay 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Cross Entropy Loss")
	plt.grid(True)
	plt.show()

	plt.figure()
	plt.plot(train_accuracy_list)
	plt.plot(valid_accuracy_list)
	plt.legend(['training accuracy', 'validation accuracy'], loc='upper right')
	plt.title("2.2.2 Accuracy for best learning rate 0.0001 and weight decay 0.001")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True)
	plt.show()
	

plot_bestLR()
#tune()









