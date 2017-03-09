import tensorflow as tf

from models.utils.tf_helpers import create_h_weights, mlp_neuron


def px_given_zy(y, z, latent_dim, num_classes, hidden_dim, input_dim, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_x', 'decoder', [num_classes + latent_dim, hidden_dim])
        w_h2, b_h2 = create_h_weights('h2_x', 'decoder', [hidden_dim, hidden_dim])

        w_mu, b_mu = create_h_weights('mu', 'decoder', [hidden_dim, input_dim])

        h1 = mlp_neuron(tf.concat([y, z], axis=1), w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)

        x_mu = tf.nn.sigmoid(tf.add(tf.matmul(h2, w_mu), b_mu))
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
        return x_mu
