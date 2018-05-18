import tensorflow as tf
import numpy as np

rand_array = np.random.rand(3, 3)
print(rand_array)
print(np.shape(rand_array))
# image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
# output: (1, ?, ?, 3)
image = tf.placeholder(tf.float32, shape=(2,2))
# output: (2, 2)
print(image.shape)
