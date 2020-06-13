import tensorflow as tf

# Model parameters
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Output value 
y = tf.placeholder(tf.float32)


# finds the best value for W, b
# FInding the least square error and finding the best regression line
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0,-1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # Evaluate accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    
    print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss))


