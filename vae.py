import time
from datetime import timedelta

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Global Dictionary of Flags
FLAGS = {
    'data_directory': 'data/MNIST/',
    'summaries_dir': 'summaries/'
}

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
## Load Data
data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

encoder_h_dim = 500
decoder_h_dim = 500
latent_dim = 2
input_dim = 784

n_classes = 10
batch_size = 100

# Height and width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


def create_biases(shape):
    return tf.Variable(tf.random_normal(shape))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# ### Helper-function for creating a new Convolutional Layer
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


##Build Model

# Encoder Model
W_encoder_h = create_weights([input_dim, encoder_h_dim])
b_encoder_h = create_biases([encoder_h_dim])

W_enconder_h_mu = create_weights([encoder_h_dim, latent_dim])
b_enconder_h_mu = create_biases([latent_dim])

W_enconder_h_var = create_weights([encoder_h_dim, latent_dim])
b_enconder_h_var = create_biases([latent_dim])
encoder_h = tf.nn.relu(tf.add(tf.matmul(x, W_encoder_h), b_encoder_h))

logvar_encoder = tf.add(tf.matmul(encoder_h, W_enconder_h_var), b_enconder_h_var)
mu_encoder = tf.add(tf.matmul(encoder_h, W_enconder_h_mu), b_enconder_h_mu)
epsilon_encoder = tf.random_normal(tf.shape(latent_dim), name='epsilon')

std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.mul(std_encoder, epsilon_encoder)

# Decoder Model
W_decoder_h = create_weights(tf.shape(latent_dim, decoder_h_dim))
b_decoder_h = create_biases([decoder_h_dim])

W_decoder_h_mu = create_weights([decoder_h_dim, input_dim])
b_decoder_h_mu = create_biases([latent_dim])

W_decoder_h_var = create_weights([decoder_h_dim, input_dim])
b_decoder_h_var = create_biases([input_dim])

decoder_h = tf.nn.relu(tf.add(tf.matmul(z, W_decoder_h), b_decoder_h))
logvar_decoder = tf.add(tf.matmul(decoder_h, W_decoder_h_var), b_decoder_h_var)
mu_decoder = tf.add(tf.matmul(decoder_h, W_decoder_h_mu), b_decoder_h_mu)
std_decoder = tf.exp(0.5 * logvar_decoder)
epsilon_decoder = tf.random_normal(tf.shape(input_dim), name='epsilon')
x_hat = mu_decoder + tf.mul(std_decoder, epsilon_decoder)

##LOSS
regularization = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder),
                                      reduction_indices=1)
reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)

session = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                     session.graph)
test_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/test')

session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

with tf.name_scope('cost'):
    with tf.name_scope('total'):
        cost = tf.reduce_mean(regularization + reconstruction_loss)
        tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer().minimize(cost)


def train_neural_network(num_iterations):
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
            summary, cost = session.run([merged, cost], feed_dict=feed_dict_train)
            train_writer.add_summary(summary, i)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, cost))

    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


train_neural_network(10000)

