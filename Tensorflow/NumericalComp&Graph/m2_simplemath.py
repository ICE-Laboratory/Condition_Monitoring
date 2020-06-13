#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:59:42 2020

MAth implementation

@author: onur
"""

import tensorflow as tf
# need to disable eager in TF2.x
#tf.compat.v1.disable_eager_execution() 
a = tf.constant(6.5, name = 'constant_a')

b = tf.constant(3.4, name = 'constant_b')
c = tf.constant(3.0, name = 'constant_c')

d = tf.constant(100.2, name = 'constant_d')

square = tf.square(a, name = 'square_a')
power = tf.pow(b, c, name = 'pow_b_c')
sqrt = tf.sqrt(d, name='sqrt_d')

final_sum = tf.add_n([square, power, sqrt], name = 'final_sum')

sess = tf.Session()



print("Square of a: ", sess.run(square))
print("Power of b ^ c: ", sess.run(power))


another_sum = tf.add_n([a,b,c,d, power], name= 'another_sum')
writer = tf.summary.FileWriter('./m2_example2', sess.graph)
#writer and sessions are handler to sources and they should be closed
writer.close()
sess.close()

"""
To see the tensor graph the command line:
    tensorboard --logdir="m2_example2"
"""
