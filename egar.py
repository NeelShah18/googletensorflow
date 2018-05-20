from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
'''
Let's start with tensorflow egar API. It allows
to executed immediately as they are called from
python. It also support to use exact same code 
that constructs tensorflow graphs can be executed
imperatively by suing eagar execution.
'''
#Starting egar mode
tfe.enable_eager_execution()
#defining constatn
a = tf.constant(2)
b = tf.constant(3)
#use variable of tensorflow without starting session
print("a: %i"%a)
print("b: %i"%b)
s = a+b
m = a*b
print("Sum: %i"%s)
print("MulL %i"%m)

#Companibility with Numpy
m1 = tf.constant([[2., 1.],
				  [1., 0.,]],dtype=tf.float32)
m2 = np.array([[3., 1.],
				[2., 3.]],dtype=np.float32)
ans = tf.matmul(m1,m2)
print("%s"%ans)

