'''
ECE521 Assignment1 02-02-2018
Part3
Yilin Chen   1000311281
Wenyu Mao    1000822292
Jiaying Wang 1000337502

'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from PIL import Image

def euclid_distance(X, Y):
	X = (tf.expand_dims(X, 1))#shape(3,1,2)	
	Y = (tf.expand_dims(Y, 0))#shape(1,3,2)
	#shape(3, 3, 2) due to broadcasting
	square_diff = tf.squared_difference(X, Y)
	return tf.reduce_sum(square_diff, 2)
    
def predict_class(row_test,train,train_target,k):

    distance_matrix=euclid_distance(row_test,train) #rowtest=1x1024 train=747*1024
    d_nearest = tf.negative(distance_matrix) #the largest negative is the nearest
    distance, indices = tf.nn.top_k(d_nearest, k) #the kth nearest
    
    #indices=(tf.expand_dims(indices, 1))
    t_target=tf.gather_nd(tf.reshape(train_target,[-1]),tf.transpose(indices))
    #t_target=tf.reshape(train_target,[-1]) #t_target=tf.reduce_sum(t_target,2)
    
    #nearestK = tf.gather_nd(t_target,Index)
	
    majority,idx,count = tf.unique_with_counts(tf.reshape(t_target,shape=[-1]))
    max_count,max_idx = tf.nn.top_k(count,k=1)
        
    #the majority train target corresponding to test data
    majority_result = tf.gather(majority,max_idx)
     
    return majority_result, tf.reshape(indices,[-1])

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

def face_recognition(task):
    trainData, validData, testData, trainTarget, validTarget, testTarget \
            = data_segmentation("data.npy", "target.npy", task)
    print np.shape(trainTarget.reshape(1,-1))
    print np.shape(validData[1,:].reshape(1,-1))
    #return
    train = tf.placeholder(tf.float32, [None, None], name="train")
    train_target = tf.placeholder(tf.float32, [None, None], name="train_target")
    valid = tf.placeholder(tf.float32, [None, None], name="valid")
    valid_target = tf.placeholder(tf.float32, [None, None], name="valid_target")
    test = tf.placeholder(tf.float32, [None, None], name="test")
    test_target = tf.placeholder(tf.float32, [None, None], name="test_target")
    K = tf.placeholder("int32", name="k")
    
    row_prediction = predict_class(test,train,train_target,K)

    sess = tf.InteractiveSession()

    K_values = [1,5,10, 25, 50, 100, 200]
    k_errors = []
    # run validation dataset
    for k in K_values:
        results = []
        #validData_result = predict_class(valid,train,train_target,k)
        for row_index in range(np.shape(validData)[0]):
            row_result, indices = sess.run(row_prediction, feed_dict={ test:validData[row_index,:].reshape(1,-1),train:trainData, train_target: trainTarget.reshape(1,-1), K:k})

            if row_result-validTarget[row_index] != 0:
                results.append(1)
            else:
                results.append(0)
            #print("result = %d, valid= %d\n"%(np.,validTarget[row_index]))
	#
        error = np.sum(results)
        k_errors.append(error)
		   	
        print("For k = %d, accuracy of valid data is %f \n"%(k, 1-error/float(np.shape(validData)[0]))) 
    
    k_best = K_values[np.argmin(k_errors)]
    print("best k is k=%d\n"%(k_best))

    # run test dataset with k_best
    results = []
    for row_index in range(np.shape(testData)[0]):
            row_result, indices = sess.run(row_prediction, feed_dict={ test:testData[row_index,:].reshape(1,-1),train:trainData, train_target: trainTarget.reshape(1,-1), K:k_best})

            if row_result-testTarget[row_index] != 0:
                results.append(1)
            else:
                results.append(0)
            #for i in range(np.shape(indices)[0]):
             #   print("i = %d\n"%(indices[i]))

    error = np.sum(results)

    print("Using k=%d, accuracy of test data is %f\n"%(k_best, 1-error/float(np.shape(testData)[0])))

    # run the test dataset with k=10
    results = []
    for row_index in range(np.shape(testData)[0]):
            row_result, indices = sess.run(row_prediction, feed_dict={ test:testData[row_index,:].reshape(1,-1),train:trainData, train_target: trainTarget.reshape(1,-1), K:10})

            if row_result-testTarget[row_index] != 0:
		if task==0:
		        target = plt.imshow(testData[row_index,:].reshape(32,32) ,cmap='gray')
			plt.title('Name Failure Test Image')
		        plt.show()
		else:
		        target = plt.imshow(testData[row_index,:].reshape(32,32),cmap='gray' )
			plt.title('Gender Failure Test Image')
		        plt.show()
                for i in range(10):
			if task==0:
		              target = plt.imshow(trainData[indices[i],:].reshape(32,32),cmap='gray')
		              plt.title('10 Nearest Name Failure Train Image')
		              plt.show()
			else:
		              target = plt.imshow(trainData[indices[i],:].reshape(32,32),cmap='gray')
		              plt.title('10 Nearest Gender Failure Train Image')
		              plt.show()
                break

face_recognition(0)
face_recognition(1)


