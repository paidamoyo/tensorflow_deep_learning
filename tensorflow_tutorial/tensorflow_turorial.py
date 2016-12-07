# https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 55,000 data points of training data (mnist.train),
# 10,000 points of test data (mnist.test),
# 5,000 points of validation data (mnist.validation).
# Both the training set and test set contain images and their corresponding labels;
# for example the training images are mnist.train.images and the training labels are mnist.train.labels.
# Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:

# The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784].
# For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print("shape of mnist: {}".format())

# We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector.
x = tf.placeholder(tf.float32, [None, 784])

# Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it
# to produce 10-dimensional vectors of evidence for the difference classes.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# To implement cross-entropy we need to first add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

# Then we can implement the cross-entropy function,
# Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter.
# Finally, tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm
# with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

#tensorFlow relies on a highly efficient C++ backend to do its computation.
# The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.
sess = tf.Session()
sess.run(init)

# Let's train -- we'll run the training step 1000 times!
# Each step of the loop, we get a "batch" of one hundred random data points from our training set.
#  We run train_step feeding in the batches data to replace the placeholders.
# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent.
# Ideally, we'd like to use all our data for every step of training because that would give us a better sense of
# what we should be doing, but that's expensive. So, instead, we use a different subset every time.
# Doing this is cheap and has much of the same benefit.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and
# then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
