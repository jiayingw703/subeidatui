import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import copy

#2.1 logistic

def squared_error(labels=None, logits=None):
	return 0.5 * (tf.multiply(tf.add(logits,-labels),tf.add(logits,-labels)))

def plot():
	y_pred = np.linspace(0,1,1000)
	y_dummy = np.zeros(shape=y_pred.shape)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	cross = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_dummy, logits=y_pred))
	squared = sess.run(squared_error(labels=y_dummy,logits=y_pred))
	plt.figure()
	plt.plot(y_pred,squared)
	#plt.plot(trainData,trainTarget,'.')
	plt.plot(y_pred,cross)#,'k', label = "Learning Rate: 0.005")
	#, 'k--', label = "Learning Rate: 0.001")
	#plt.plot(training_loss_3)#, 'k-', label = "Learning Rate: 0.0001")
	#legend = ax.legend(loc="upper center", shadow=False)
	#frame = legend.get_frame()
	#frame.set_facecolor('0.90')
	plt.legend(['Squared-error Loss','Cross Entropy'], loc='upper right')
	plt.title("2.1.3 Squared-error Loss VS Cross Entropy Loss")
	plt.ylabel('Loss')
	#plt.xlim([90,100])
	plt.grid(True)
	plt.xlabel('y_pred')
	#plt.title("Learning rate: %f"%learning_rate)
	plt.show()
	
plot()
