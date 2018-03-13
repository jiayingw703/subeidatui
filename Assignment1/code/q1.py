'''
ECE521 Assignment1 02-02-2018
Part1
Yilin Chen   1000311281
Wenyu Mao    1000822292
Jiaying Wang 1000337502

'''


import tensorflow as tf
import numpy as np

def euclid_distance(X, Y):
	X = (tf.expand_dims(X, 1))#shape(4,1,2)	
	Y = (tf.expand_dims(Y, 0))#shape(1,3,2)
	#shape(4, 3, 2) due to broadcasting
	square_diff = tf.squared_difference(X, Y)
	return tf.reduce_sum(square_diff, 2)

#for self testing
X = tf.Variable([[1,2],[3,4],[5,6],[7,8]])
Y = tf.Variable([[1,1],[2,2],[3,3]])

result = euclid_distance(X, Y) #shape(4, 3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run("dhuwe" +result)
