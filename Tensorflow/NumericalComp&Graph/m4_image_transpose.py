import tensorflow as tf
import matplotlib.image as mp_img
from matplotlib import pyplot as plot
import os

filename = "./gregHume_dandelion.jpg"

image = mp_img.imread(filename)

#[width, height, rbg]
print("Image shape: ", image.shape)
# 
print("Image array: ", image)

plot.imshow(image)
plot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # flip the height and width
    # swap the first and second axis around the original axis
    #transpose = tf.transpose(x, perm=[1, 0, 2])
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)

    print("Transposed image shape: ", result.shape)
    plot.imshow(result)
    plot.show()
    



