import tensorflow as tf

x1 = tf.constant(3)
x2 = tf.constant(2)
result = tf.multiply(x1,x2)
sess = tf.Session()
print(sess.run(result))
