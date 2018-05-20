import tensorflow as tf
import numpy as np
import pandas as pd
import tkinter
import matplotlib.pyplot as plt
rng = np.random

learning_rate = 0.02
training_epochs = 10000
display_step = 100

data = pd.read_csv("early-senate-polls.csv")

train_y = np.array(data["election_result"])
train_x = np.array(data["presidential_approval"])

n_samples = train_x.shape[0]

x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
#w = tf.Variable(np.random(), name="weight")
#b = tf.Variable(np.random(), name="bias")

pred = tf.add(tf.multiply(x,w), b)

cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		for(p,q) in zip(train_x, train_y):
			sess.run(optimizer, feed_dict={x: p, y: q})

		if (epoch+1)%display_step==0:
			c = sess.run(cost, feed_dict={x:train_x, y:train_y})
			print("Epoch:", '%04d' %(epoch+1), "cost=","{:.9f}".format(c),"w=",sess.run(w),"b=",sess.run(b))
	print("Optimization Finished!")
	training_cost = sess.run(cost, feed_dict={x:train_x, y:train_y})
	print("Training cost=", training_cost,"w=",sess.run(w),"b=",sess.run(b))

	plt.plot(train_x, train_y, 'ro', label='Original data')
	plt.plot(train_x, sess.run(w)*train_x+sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()