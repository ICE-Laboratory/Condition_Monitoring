import tensorflow as tf

#multi dimensional tensor setup
x = tf.constant([100,200,300], name='x')
y = tf.constant([1,2,3], name='y')

#add all the elements in one tensor
sum_x = tf.reduce_sum(x, name='sum_x')
#multiply each elements in tensor
prod_y = tf.reduce_prod(y, name='prod_y')

final_div =  tf.div( sum_x, prod_y, name='final_mean')
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sess = tf.Session()
writer = tf.summary.FileWriter('./m2_example4', sess.graph)

print("x: ",sess.run(x))
print("y: ", sess.run(y))

print("sum(x): ", sess.run(sum_x))
print("prod(y): ", sess.run(prod_y))
print("sum(x) / prod(y): ", sess.run(final_div))
print("mean(sum(x), prod(y)):", sess.run(final_mean))

writer.close()
sess.close()

