import tensorflow as tf

#Defining constant using tensorflow object
a = tf.constant(2)
b = tf.constant(3)

'''
Open tensorflow session and perform the task, Here we use "with" open the tensorflow because with will close the session automatically so we dont need to remember to close 
each sessiona fter starting it. We can use those constatn variable as python variable and perform the task or we can use tensorflow inbuild function to perform mathematical task.
Bdw launching the session means define basic graph!!!!
'''
with tf.Session() as sess:
	print("A is %i"%sess.run(a))
	print("B is %i"%sess.run(b))
	print("Addition is: %i"%sess.run(a+b))
	print("Multiplication is: %i"%sess.run(a*b))

'''
Here, placeholder works like input of the graph. Means it defines what will be input for current runing session
'''
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

'''
As we can see a and b is now placeholder means input for current runing session. 
'''
with tf.Session() as sess:
	print("Addition: %i"%sess.run(add, feed_dict={a:10, b:15}))
	print("Multiplication: %i"%sess.run(mul, feed_dict={a:5, b:6}))

'''
Creating two constatn matrix m1=1*2 and m2=2*1
'''
m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[2.], [2.]])

#Deafult function of tensorflow to do matrix multiplication. Here object is created name prod which perform matrix multiplication of m1 and m2
prod = tf.matmul(m1,m2)

#This session print the muatrix multiplication
with tf.Session() as sess:
	result = sess.run(prod)
	print(result)
