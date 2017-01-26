import math
import os
import time
from datetime import timedelta

import cifar10
import matplotlib.pyplot as plt
import numpy as np
import prettytensor as pt
import tensorflow as tf
from cifar10 import img_size, num_channels, num_classes
from sklearn.metrics import confusion_matrix

# cifar10.data_path = "data/CIFAR-10/"
# The CIFAR-10 data-set is about 163 MB and will be downloaded automatically if it is not located in the given path.
cifar10.maybe_download_and_extract()

# Load the class-names.
class_names = cifar10.load_class_names()
print(class_names)

# Load the training-set. This returns the images, the class-numbers as integers, and the class-numbers as One-Hot encoded arrays called labels.
images_train, cls_train, labels_train = cifar10.load_training_data()

# Load the test-set.
images_test, cls_test, labels_test = cifar10.load_test_data()

# The CIFAR-10 data-set has now been loaded and consists of 60,000 images and associated labels (i.e. classifications of the images). The data-set is split into 2 mutually exclusive sub-sets, the training-set and the test-set.
print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# ### Data Dimensions

# The data dimensions are used in several places in the source-code below. They have already been defined in the cifar10 module, so we just need to import them.
# The images are 32 x 32 pixels, but we will crop the images to 24 x 24 pixels.

img_size_cropped = 24


# ### Helper-function for plotting images

# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.

    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)

        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}, Pred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Plot a few images to see if data is correct
# Get the first images from the test-set.
images = images_test[0:9]

# Get the true classes for those images.
cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=False)

# The pixelated images above are what the neural network will get as input.
# The images might be a bit easier for the human eye to recognize if we smoothen the pixels.
plot_images(images=images, cls_true=cls_true, smooth=True)

# ## TensorFlow Graph

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)


# ### Helper-function for creating Pre-Processing
# The pre-processing is different for training and testing of the neural network:
# * For training, the input images are randomly cropped, randomly flipped horizontally,
# and the hue, contrast and saturation is adjusted with random values.
# This artificially inflates the size of the training-set by creating random variations of the original input images.
# Examples of distorted images are shown further below.
# 
# * For testing, the input images are cropped around the centre and nothing else is adjusted.


def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


# The function above is called for each image in the input batch using the following function.

# In[20]:

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images


# In order to plot the distorted images, we create the pre-processing graph for TensorFlow, so we may execute it later.
distorted_images = pre_process(images=x, training=True)


# ### Helper-function for creating Main Processing

# The following helper-function creates the main part of the convolutional neural network.
#  It uses Pretty Tensor which was described in the previous tutorials.

# In[22]:

def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True) \
            .max_pool(kernel=2, stride=2) \
            .conv2d(kernel=5, depth=64, name='layer_conv2') \
            .max_pool(kernel=2, stride=2) \
            .flatten() \
            .fully_connected(size=256, name='layer_fc1') \
            .fully_connected(size=128, name='layer_fc2') \
            .softmax_classifier(num_classes, labels=y_true)

    return y_pred, loss


# ### Helper-function for creating Neural Network

# The following helper-function creates the full neural network, which consists of the pre-processing and main-processing defined above.
# 
# Note that the neural network is enclosed in the variable-scope named 'network'.
# This is because we are actually creating two neural networks in the TensorFlow graph.
# By assigning a variable-scope like this, we can re-use the variables for the two neural networks,
# so the variables that are optimized for the training-network are re-used for the other network that is used for testing.

#
def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


# ### Create Neural Network for Training Phase

# First create a TensorFlow variable that keeps track of the number of optimization iterations performed so far.
#  In the previous tutorials this was a Python variable, but in this tutorial we want to save this variable with all
#  the other TensorFlow variables in the checkpoints.
# Note that `trainable=False` which means that TensorFlow will not try to optimize this variable.

global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

# Create the neural network to be used for training. The `create_network()` function returns both `y_pred` and `loss`,
#  but we only need the `loss`-function during training.

_, loss = create_network(training=True)

# Create an optimizer which will minimize the `loss`-function.
# Also pass the `global_step` variable to the optimizer so it will be increased by one after each iteration.

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# ### Create Neural Network for Test Phase / Inference

# Now create the neural network for the test-phase. Once again the `create_network()`
# function returns the predicted class-labels `y_pred` for the input images, as well as the `loss`-function to be
#  used during optimization. During testing we only need `y_pred`.

y_pred, _ = create_network(training=False)

# We then calculate the predicted class number as an integer.
# The output of the network `y_pred` is an array with 10 elements.
#  The class number is the index of the largest element in the array.

y_pred_cls = tf.argmax(y_pred, dimension=1)

# Then we create a vector of booleans telling us whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# The classification accuracy is calculated by first type-casting the vector of booleans to floats,
# so that False becomes 0 and True becomes 1, and then taking the average of these numbers.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ### Saver
# 
# In order to save the variables of the neural network, so they can be reloaded quickly without having to train the
# network again, we now create a so-called Saver-object which is used for storing and retrieving all
# the variables of the TensorFlow graph. Nothing is actually saved at this point, which will be done further below.
saver = tf.train.Saver()


# ### Getting the Weights

# Further below, we want to plot the weights of the neural network. When the network is constructed using Pretty Tensor,
#  all the variables of the layers are created indirectly by Pretty Tensor. We therefore have to retrieve the variables
# from TensorFlow.We used the names `layer_conv1` and `layer_conv2` for the two convolutional layers.
# These are also called variable scopes. Pretty Tensor automatically gives names to the variables it creates for each layer,
# so we can retrieve the weights for a layer using the layer's scope-name and the variable-name.
# The implementation is somewhat awkward because we have to use the TensorFlow function `get_variable()`
# which was designed for another purpose; either creating a new variable or re-using an existing variable.
#  The easiest thing is to make the following helper-function.
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


# Using this helper-function we can retrieve the variables.
# These are TensorFlow objects. In order to get the contents of the variables,
# you must do something like: `contents = session.run(weights_conv1)` as demonstrated further below.
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')


# ### Getting the Layer Outputs

# Similarly we also need to retrieve the outputs of the convolutional layers.
#  The function for doing this is slightly different than the function above for getting the weights.
# Here we instead retrieve the last tensor that is output by the convolutional layer.
def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor


# Get the output of the convoluational layers so we can plot them later.
output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

# ## TensorFlow Run
# ### Create TensorFlow session
# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.
session = tf.Session()

# ### Restore or initialize variables
# Training this neural network may take a long time, especially if you do not have a GPU.
#  We therefore save checkpoints during training so we can continue training at another time (e.g. during the night),
#  and also for performing analysis later without having to train the neural network every time we want to use it.
# If you want to restart the training of the neural network, you have to delete the checkpoints first.
# 
# This is the directory used for the checkpoints.
save_dir = 'checkpoints/'

# Create the directory if it does not exist.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# This is the base-filename for the checkpoints, TensorFlow will append the iteration number, etc.
save_path = save_dir + 'cifar10_cnn'

# First try to restore the latest checkpoint.
# This may fail and raise an exception e.g. if such a checkpoint does not exist, or if you have changed the TensorFlow graph.
try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

# ### Helper-function to get a random training-batch

# There are 50,000 images in the training-set.
#  It takes a long time to calculate the gradient of the model using all these images.
# We therefore only use a small batch of images in each iteration of the optimizer.
# If your computer crashes or becomes very slow because you run out of RAM,
# then you may try and lower this number, but you may then need to perform more optimization iterations.

train_batch_size = 64


# Function for selecting a random batch of images from the training-set.
def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


# ### Helper-function to perform optimization

# This function performs a number of optimization iterations so as to gradually improve
# the variables of the network layers. In each iteration, a new batch of data is selected from
# the training-set and then TensorFlow executes the optimizer using those training samples.
#   The progress is printed every 100 iterations. A checkpoint is saved every 1000 iterations and also after the last iteration.

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Helper-function to plot example errors
# Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# ### Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


# ### Helper-functions for calculating classifications
# This function calculates the predicted classes of images and also returns a boolean array whether
#  the classification of each image is correct. The calculation is done in batches because it might use
# too much RAM otherwise. If your computer crashes then you can try and lower the batch-size.

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


# Calculate the predicted class for the test-set.
def predict_cls_test():
    return predict_cls(images=images_test,
                       labels=labels_test,
                       cls_true=cls_test)


# ### Helper-functions for the classification accuracy
# This function calculates the classification accuracy given a boolean array
# whether each image was correctly classified.
#  E.g. `classification_accuracy([True, True, False, False, False]) = 2/5 = 0.4`.
# The function also returns the number of correct classifications.


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


# ### Helper-function for showing the performance
# Function for printing the classification accuracy on the test-set.
# It takes a while to compute the classification for all the images in the test-set,
# that's why the results are re-used by calling the above functions directly from this function,
#  so the classifications don't have to be recalculated by each function.

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# ### Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Helper-function for plotting the output of convolutional layers
def plot_layer_output(layer_output, image):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {x: [image]}

    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ## Examples of distorted input images
# In order to artificially inflate the number of images available for training,
# the neural network uses pre-processing with random distortions of the input images.
# This should hopefully make the neural network more flexible at recognizing and classifying images.
# This is a helper-function for plotting distorted input images.

def plot_distorted_image(image, cls_true):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {x: image_duplicates}

    # Calculate only the pre-processing of the TensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))


# Helper-function for getting an image and its class-number from the test-set.
def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]


# Get an image and its true class from the test-set.
img, cls = get_test_image(16)

# Plot 9 random distortions of the image. If you re-run this code you will get slightly different results.
plot_distorted_image(img, cls)

# ## Perform optimization
# Because we are saving the checkpoints during optimization,
# and because we are restoring the latest checkpoint when restarting the code, we can stop and continue the optimization later.
if False:
    optimize(num_iterations=10000)

# ## Results
# After 150,000 optimization iterations, the classification accuracy is about 79-80% on the test-set.
# Examples of mis-classifications are plotted below. Some of these are difficult to recognize even for
#  humans and others are reasonable mistakes e.g. between a large car and a truck, or between a cat and a dog,
#  while other mistakes seem a bit strange.

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ### Convolutional Weights
# 
# The following shows some of the weights (or filters) for the first convolutional layer.
# There are 3 input channels so there are 3 of these sets, which you may plot by changing the `input_channel`.
# Note that positive weights are red and negative weights are blue.

plot_conv_weights(weights=weights_conv1, input_channel=0)

# Plot some of the weights (or filters) for the second convolutional layer.
# These are apparently closer to zero than the weights for the first convolutional layers, see the lower standard deviation.

plot_conv_weights(weights=weights_conv2, input_channel=1)


# ### Output of convolutional layers
# Helper-function for plotting an image.

def plot_image(image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels.
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Plot an image from the test-set. The raw pixelated image is used as input to the neural network.

img, cls = get_test_image(16)
plot_image(img)

# Use the raw image as input to the neural network and plot the output of the first convolutional layer.

plot_layer_output(output_conv1, image=img)

# Using the same image as input to the neural network, now plot the output of the second convolutional layer.

plot_layer_output(output_conv2, image=img)

# ### Predicted class-labels
# Get the predicted class-label and class-number for this image.
label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                   feed_dict={x: [img]})

# Print the predicted class-label.

# Set the rounding options for numpy.
np.set_printoptions(precision=3, suppress=True)

# Print the predicted label.
print(label_pred[0])

# The predicted class-label is an array of length 10,
# with each element indicating how confident the neural network is that the image is the given class.
# 
# In this case the element with index 3 has a value of 0.493, while the element with index 5 has a value of 0.490.
# This means the neural network believes the image either shows a class 3 or class 5, which is a cat or a dog, respectively.
print(class_names[3])

print(class_names[5])

# ## Close TensorFlow Session

session.close()  # ## Conclusion

# ## License (MIT)
# 
# Copyright (c) 2016 by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
