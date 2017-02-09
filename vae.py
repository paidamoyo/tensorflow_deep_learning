import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Global Dictionary of Flags
FLAGS = {
    'data_directory': 'data/MNIST/',
    'summaries_dir': 'summaries/',
    'save_path': 'results/results/train_weights',
}

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

encoder_h_dim = 500
decoder_h_dim = 500
latent_dim = 200
img_size = 28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# ### Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


def create_biases(shape):
    return tf.Variable(tf.random_normal(shape))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# ### Helper-function for creating a new Convolutional Layer
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


##Build Model
def recognition_network():
    # Variables
    W_encoder_h_1 = create_weights([img_size_flat, encoder_h_dim])
    b_encoder_h_1 = create_biases([encoder_h_dim])
    variable_summaries(W_encoder_h_1, 'W_encoder_h_1')
    variable_summaries(b_encoder_h_1, 'b_encoder_h_1')

    W_encoder_h_2 = create_weights([encoder_h_dim, encoder_h_dim])
    b_encoder_h_2 = create_biases([encoder_h_dim])
    variable_summaries(W_encoder_h_2, 'W_encoder_h_2')
    variable_summaries(b_encoder_h_2, 'b_encoder_h_2')

    W_enconder_h_mu = create_weights([encoder_h_dim, latent_dim])
    b_enconder_h_mu = create_biases([latent_dim])
    variable_summaries(W_enconder_h_mu, 'W_enconder_h_mu')
    variable_summaries(b_enconder_h_mu, 'b_enconder_h_mu')

    W_enconder_h_var = create_weights([encoder_h_dim, latent_dim])
    b_enconder_h_var = create_biases([latent_dim])
    variable_summaries(W_enconder_h_var, 'W_enconder_h_var')
    variable_summaries(b_enconder_h_var, 'b_enconder_h_var')

    # Hidden layers
    encoder_h_1 = tf.nn.relu(tf.add(tf.matmul(x, W_encoder_h_1), b_encoder_h_1))
    encoder_h_2 = tf.nn.relu(tf.add(tf.matmul(encoder_h_1, W_encoder_h_2), b_encoder_h_2))

    # latent layer mu and var
    encoder_logvar = tf.add(tf.matmul(encoder_h_2, W_enconder_h_var), b_enconder_h_var)
    encoder_mu = tf.add(tf.matmul(encoder_h_1, W_enconder_h_mu), b_enconder_h_mu)

    # latent layer
    epsilon_encoder = tf.random_normal(tf.shape(latent_dim), name='epsilon')
    std_encoder = tf.exp(0.5 * encoder_logvar)
    z = encoder_mu + tf.mul(std_encoder, epsilon_encoder)

    # regularization loss
    regularization = -0.5 * tf.reduce_sum(1 + encoder_logvar - tf.pow(encoder_mu, 2) - tf.exp(encoder_logvar),
                                          reduction_indices=1)
    return z, regularization


# Encoder Model
z, regularization_loss = recognition_network()


def generator_network():
    # Variables
    W_decoder_h_1 = create_weights([latent_dim, decoder_h_dim])
    b_decoder_h_1 = create_biases([decoder_h_dim])
    variable_summaries(W_decoder_h_1, 'W_decoder_h_1')
    variable_summaries(b_decoder_h_1, 'b_decoder_h_1')

    W_decoder_h_2 = create_weights([decoder_h_dim, decoder_h_dim])
    b_decoder_h_2 = create_biases([decoder_h_dim])
    variable_summaries(W_decoder_h_2, 'W_decoder_h_2')
    variable_summaries(b_decoder_h_2, 'b_decoder_h_2')

    W_decoder_r = create_weights([decoder_h_dim, img_size_flat])
    b_decoder_r = create_biases([img_size_flat])
    variable_summaries(W_decoder_r, 'W_decoder_r')
    variable_summaries(b_decoder_r, 'b_decoder_r')

    # Decoder hidden layer
    decoder_h_1 = tf.nn.relu(tf.add(tf.matmul(z, W_decoder_h_1), b_decoder_h_1))
    decoder_h_2 = tf.nn.relu(tf.add(tf.matmul(decoder_h_1, W_decoder_h_2), b_decoder_h_2))

    # Reconstrunction layer
    x_hat = tf.add(tf.matmul(decoder_h_2, W_decoder_r), b_decoder_r)
    tf.summary.image('x_hat', tf.reshape(x_hat[0], [1, 28, 28, 1]))

    return x_hat


# Decoder Model
x_hat = generator_network()

reconstruction_loss = tf.reduce_sum(tf.squared_difference(x_hat, x), reduction_indices=1)

loss = tf.reduce_mean(regularization_loss + reconstruction_loss)
tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                     session.graph)
train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

## SAVER
saver = tf.train.Saver()


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for step in range(num_iterations):

        total_iterations += 1

        x_batch, _ = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch}

        summary, cur_loss, _ = session.run([merged, loss, optimizer], feed_dict=feed_dict_train)
        train_writer.add_summary(summary, step)

        if total_iterations % 100 == 0:
            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=session, save_path=FLAGS['save_path'])
            print("Optimization Iteration: {}, Training Loss: {}".format(step + 1, cur_loss))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def reconstruct(x_test):
    return session.run(x_hat, feed_dict={x: x_test})


def plot_images(x_test, x_reconstruct):
    assert len(x_test) == 5

    plt.figure(figsize=(8, 12))
    for i in range(5):
        # Plot image.
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("reconstructed digit")


def test_reconstruction():
    saver.restore(sess=session, save_path=FLAGS['save_path'])
    x_test = mnist.test.next_batch(100)[0][0:5, ]
    print(np.shape(x_test))
    x_reconstruct = reconstruct(x_test)
    plot_images(x_test, x_reconstruct)


train_neural_network(10000)
# test_reconstruction()
session.close()
