import tensorflow as tf

from VAE.utils.settings import initialize
from VAE.utils.tf_helpers import create_h_weights, mlp_neuron

FLAGS = initialize()


def px_given_z1(z1, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_x', 'decoder', [FLAGS['latent_dim'], FLAGS['m1_h_dim']])
        w_h2, b_h2 = create_h_weights('h2_x', 'decoder', [FLAGS['m1_h_dim'], FLAGS['m1_h_dim']])

        w_mu, b_mu = create_h_weights('mu', 'decoder', [FLAGS['m1_h_dim'], FLAGS['input_dim']])
        # Model
        # Decoder hidden layer
        h1 = mlp_neuron(z1, w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)
        # Reconstruction layer
        # x_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)
        x_mu = tf.nn.sigmoid(tf.add(tf.matmul(h2, w_mu), b_mu))
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
        return x_mu
