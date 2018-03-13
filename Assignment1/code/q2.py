'''
ECE521 Assignment1 02-02-2018
Part2
Yilin Chen   1000311281
Wenyu Mao    1000822292
Jiaying Wang 1000337502

'''


import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# generate data set
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

def euclid_distance(X, Y):
	X = (tf.expand_dims(X, 1))#shape(3,1,2)	
	Y = (tf.expand_dims(Y, 0))#shape(1,3,2)
	#shape(3, 3, 2) due to broadcasting
	square_diff = tf.squared_difference(X, Y)
	return tf.reduce_sum(square_diff, 2)

def responsibility(distance_matrix, k):
  
    #We need to index the closest values
    #We need a matrix that each row represents 80 training data correspond to 1 test point, in total, we have 10 rows of such data)
    d_trans=tf.transpose(distance_matrix)
    #Now we have a matrix has all the distance as positive number
    #tf.nn.top_k gets the kth largest
    #Since we need to get the nearest
    d_nearest = tf.negative(d_trans) #the largest negative is the nearest
    distance, indices = tf.nn.top_k(d_nearest, k) #the kth nearest
    
    #get number of training Data
    td_num=tf.shape(d_nearest)[1]
    d_nearest_range=tf.range(td_num)
    #to compare with the indices, reshape the d_nearest_range in order to broadcasting later
    index_compare=tf.reshape(d_nearest_range,[1,1,-1]) #1 1 4

    #in order to broadcast, we should change indices's last dimension to 1
    indices=tf.expand_dims(indices,2)#3 2 1
    
    #find the indices for the nearest distance
    nearest_index=tf.reduce_sum(tf.to_float(tf.equal(index_compare,indices)),1) 
    tf.cast(nearest_index,tf.float64)
    result=nearest_index/tf.to_float(k)
    tf.cast(result,tf.float64)
    return result
#######################responsibility done#####################
def prediction(train,test,train_target,k):
    #get reposibility 
    distance_matrix=euclid_distance(train,test)
    r_star=tf.transpose(tf.cast(responsibility(distance_matrix,k),tf.float64))
    prediction_y=tf.matmul(tf.transpose(train_target),r_star)
    return tf.transpose(prediction_y)

def MSE(train,test,train_target,test_target,k):
    #get prediction
    predicted_result = prediction(train,test,train_target,k)
    #get squared sum and cast the type
    squared_sum=tf.reduce_sum(tf.pow(tf.to_float(tf.subtract(predicted_result, test_target)),2)) 
    num_test = tf.to_float(tf.shape(test)[0])
    result = squared_sum/(num_test*2)
    return result


############################for ploting#########################
def get_mse(Data, Target, size, k):
  train_X=tf.placeholder(tf.float64, [80, 1])#X
  target_Y=tf.placeholder(tf.float64, [80, 1])#Y
  x_star=tf.placeholder(tf.float64, [None, 1])#x_T
  new_y=tf.placeholder(tf.float64, [None, 1])#y_T
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  mse=MSE(train_X,x_star,target_Y,new_y,k)
  return sess.run(mse, feed_dict={train_X:trainData, x_star: Data, target_Y: trainTarget, new_y: Target})

def plot():
  train_X=tf.placeholder(tf.float64, [80, 1])#X
  target_Y=tf.placeholder(tf.float64, [80, 1])#Y
  x_star=tf.placeholder(tf.float64, [None, 1])#x_T
  new_y=tf.placeholder(tf.float64, [None, 1])#y_T
  K = tf.placeholder("int32")
  mse=MSE(train_X,x_star,target_Y,new_y,K)
  predicted_result = prediction(train_X,x_star,target_Y,K)
  sess = tf.InteractiveSession()

  X = np.linspace(0.0,11.0,num=1000)[:,np.newaxis] # for plotting

  for k in[1,3,5,50]:
	train_mse= sess.run(mse, feed_dict={train_X:trainData, x_star: trainData, target_Y: trainTarget, new_y:trainTarget, K:k})
	print("Train MSE loss:%f, k=%d"%(train_mse, k))

  for k in[1,3,5,50]:
	test_mse= sess.run(mse, feed_dict={train_X:trainData, x_star: testData, target_Y: trainTarget, new_y:testTarget, K:k})
	print("test MSE loss:%f, k=%d"%(test_mse, k))

  MSE_values = []
  for k in[1,3,5,50]:
	valid_mse= sess.run(mse, feed_dict={train_X:trainData, x_star: validData, target_Y: trainTarget, new_y:validTarget, K:k})
	MSE_values.append(valid_mse)
	print("validation MSE loss:%f, k=%d"%(valid_mse, k))
  
  k_values = [1,3,5,50]
  k_best = k_values[np.argmin(MSE_values)]
  print("best k with is k=%d, minumum MSE is %f"%(k_best, np.min(MSE_values)))

  for k in [1,3,5,50]:
	predicted = sess.run(predicted_result, feed_dict={train_X:trainData, x_star: X, target_Y: trainTarget, K:k})
	plt.figure(k)
	plt.plot(trainData,trainTarget,'.')
	plt.plot(X,predicted, '-')
	plt.title("k-NN regression, k=%d"%k)
	plt.show()

plot() 

	







