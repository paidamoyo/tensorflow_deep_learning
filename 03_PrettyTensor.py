import time
from datetime import timedelta

import numpy as np
import prettytensor as pt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

# Global Dictionary of Flags
FLAGS = {
    'data_directory': 'data/MNIST/',
    'summaries_dir': 'summaries/',
    'num_iterations': 1
}

# Convolutional Layer 1.
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 16  # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 36  # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.

# ## Load Data
data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# ## Data Dimensions

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# ### Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


def build_network():
    ##Pretty Tensor Graph
    x_pretty = pt.wrap(x_image)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty. \
            conv2d(kernel=5, depth=16, name='layer_conv1'). \
            max_pool(kernel=5, stride=2). \
            conv2d(kernel=5, depth=36, name='layer_conv2'). \
            max_pool(kernel=5, stride=2). \
            flatten(). \
            fully_connected(size=128, name='layer_fc1'). \
            softmax_classifier(10, labels=y_true)
    return y_pred, loss


y_pred, loss = build_network()

y_pred_cls = tf.argmax(y_pred, dimension=1)

cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

accuracy = y_pred.evaluate_classifier(y_true)

session = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                     session.graph)
test_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/test')

session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            summary, acc = session.run([merged, accuracy], feed_dict=feed_dict_train)
            test_writer.add_summary(summary, i)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))
        else:
            summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    incorrect_images = data.test.images[incorrect]

    ## save image
    for i in range(0, 9):
        with tf.variable_scope("incorrect_images"):
            tf.image_summary('incorrect image/{}'.format(i), incorrect_images[i], max_images=1)


def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)
    ## save image
    with tf.variable_scope('accuracy'):
        tf.image_summary('confusion matrix', cm, max_images=1)


test_batch_size = 256


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0
    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        test_images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: test_images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# ## Performance before any optimization
print_test_accuracy()

# ## Performance after 1 optimization iteration
optimize(FLAGS['num_iterations'])

print_test_accuracy()

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


## Get weights
def get_variables(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('weights')
        # biases = tf.get_variable('biases')

    return weights


# ## Visualization of Weights and Layers
def save_conv_weights(weights, layer_name, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)
    tf.summary.image("w", w)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    for i in range(0, num_filters):
        # Get the weights for the i'th filter of the input channel.
        # See new_conv_layer() for details on the format
        # of this 4-dim tensor.
        img = w[:, :, input_channel, i]

        ## save image
        with tf.variable_scope(layer_name):
            tf.image_summary('{}/features/{}'.format(layer_name, i), img, max_images=1)


layer_conv_1 = 'layer_conv1'

save_conv_weights(get_variables(layer_conv_1), layer_conv_1)

# ### Close TensorFlow Session
session.close()
