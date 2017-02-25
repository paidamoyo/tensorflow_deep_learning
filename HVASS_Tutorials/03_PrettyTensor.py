import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import prettytensor as pt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

# Global Dictionary of Flags
FLAGS = {
    'data_directory': 'data/MNIST/',
    'summaries_dir': 'summaries/',
    'num_iterations': 10000,
    'results': 'results/',
    'save_path': 'results/best_validation',
    'test_batch_size': 256,
    'require_improvement': 1000,
    'train_batch_size': 64
}

# ## Load Data
data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# ## Data Dimensions
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# ### Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])



def variable_summaries(var, summary_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


## Get weights
def get_variables(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('weights')
        bias = tf.get_variable('bias')
        variable_summaries(bias, 'bias')
        variable_summaries(weights, 'weights')


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

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
# session = tf.Session(config=config)
session = tf.Session()

get_variables('layer_conv1')
get_variables('layer_conv2')
get_variables('layer_fc1')

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                     session.graph)
test_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/test')


def init_all_variables():
    session.run(tf.global_variables_initializer())


init_all_variables()

## SAVER
saver = tf.train.Saver()

with tf.name_scope('early_stopping'):
    best_validation_accuracy = 0.0
    # Iteration number for last improvement to validation accuracy
    last_improvement = 0
    tf.summary.scalar('best_validation_accuracy', best_validation_accuracy)
    tf.summary.scalar('last_improvement', last_improvement)

# stop optimization if no improvement found in this many iteration


# Counter for total number of iterations performed so far.
total_iterations = 0


def validation_accuracy():
    correct, _ = predict_cls(images=data.validation.images,
                             labels=data.validation.labels,
                             cls_true=convert_labels_to_cls(data.validation.labels))
    return cls_accuracy(correct)


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        total_iterations += 1

        x_batch, y_true_batch = data.train.next_batch(FLAGS['train_batch_size'])
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_train)
        train_writer.add_summary(summary, i)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy on the training-set.
            summary, acc_train = session.run([merged, accuracy], feed_dict=feed_dict_train)
            test_writer.add_summary(summary, i)

            # Calculate the accuracy
            acc_validation, _ = validation_accuracy()
            if acc_validation > best_validation_accuracy:
                # update best validation accuracy
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=FLAGS['save_path'])

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''

                # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        if total_iterations - last_improvement > FLAGS['require_improvement']:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Input Images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    incorrect_images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = convert_labels_to_cls(data.test.labels)[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    cls_true = convert_labels_to_cls(data.test.labels)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)
    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    plt.imsave(FLAGS['results'] + 'confusion_matrix', cm)


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct, cls_pred = predict_cls(images=data.test.images,
                                    labels=data.test.labels,
                                    cls_true=(convert_labels_to_cls(data.test.labels)))
    acc, correct_sum = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def convert_labels_to_cls(labels):
    return np.argmax(labels, axis=1)


def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()
    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


def predict_cls(images, labels, cls_true):
    # Number of images in the test-set.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)

        # Get the images from the test-set between index i and j.
        test_images = images[i:j, :]

        # Get the associated labels.
        labels = labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: test_images,
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


print('Performance before any optimization:')
print_test_accuracy()

# ## Train Neural Net
optimize(FLAGS['num_iterations'])

print('Performance after early stopping:')
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

init_all_variables()
saver.restore(sess=session, save_path=FLAGS['save_path'])
print('Performance for best validation iteration:')
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ### Close TensorFlow Session
session.close()
