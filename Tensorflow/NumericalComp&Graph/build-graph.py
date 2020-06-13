#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:14:07 2020

@author: onur
"""


import tensorflow as tf


#The tensorflow core r2.0 have enabled eager execution by default so doesn't need to write
#In tensor 2.0 you have to use this in order to start a session (graph)
#tf.compat.v1.disable_eager_execution()
#div =tf.compat.v1.div(c,d, name = 'div')

#setting up 4 constants
#represents the input of our computation

a = tf.constant(6, name = 'constant_a') # 6 is the value of constant
b = tf.constant(3, name = 'constant_b')
c = tf.constant(10, name = 'constant_c')
d = tf.constant(5, name = 'constant_d')

mul = tf.multiply(a,b, name='mul')
div =tf.div(c,d, name = 'div')
#the output of the mul and div will be process in down code
addn = tf.add_n([mul, div], name='addn') # summs up the ellements in array

print(addn)
#output: <tf.Tensor: shape=(), dtype=int32, numpy=20>
#addn:0 --> name of the parameter




#This is how you start a session in tensorflow 2.0
 # # Launch the graph in a session.
 # with tf.compat.v1.Session() as ses:

 #     # Build a graph.
 #     a = tf.constant(5.0)
 #     b = tf.constant(6.0)
 #     c = a * b

 #     # Evaluate the tensor `c`.
 #     print(ses.run(c))

sess = tf.Session()
sess.run(addn)

writer = tf.summary.FileWriter('./m2_example1', sess.graph)
#writer and sessions are handler to sources and they should be closed
writer.close()
sess.close()

"""
To see the tensor graph the command line:
    tensorboard --logdir="m2_example1"
"""


