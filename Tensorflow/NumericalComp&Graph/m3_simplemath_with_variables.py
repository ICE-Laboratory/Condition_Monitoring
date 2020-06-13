import tensorflow as tf

# y = Wx + b
W = tf.Variable([2.5, 4.0], tf.float32, name="var_W")

x = tf.placeholder(tf.float32, name="x")
b = tf.Variable([5.0, 10.0], tf.float32, name="var_b")

y = W * x + b

# Initialize all variables defined
"""
If you want to initialize only one variable

init = tf.variables_initializer([VARIABLE])
"""
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # this will initialize the variables
    sess.run(init)

    print("Final result: Wx + b =", sess.run(y, feed_dict={x: [10, 100]}))

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()
# it takes the result of the calculation, assigns it to the result variabl
result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result number * multiplier =", sess.run(result))
        print("Increment multiplier, new value =", sess.run(multiplier.assign_add(1)))
