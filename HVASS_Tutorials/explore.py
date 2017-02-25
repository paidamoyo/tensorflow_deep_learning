import numpy as np
import tensorflow as tf


a = tf.constant(np.array([[.1, .3, .5, .9]]))
s = tf.Session()
print(s.run(tf.nn.softmax(a)))

