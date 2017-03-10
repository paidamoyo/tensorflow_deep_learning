import tensorflow as tf

from models.utils.tf_helpers import create_nn_weights, mlp_neuron


def pz1_given_z2y(y, z, latent_dim, num_classes, hidden_dim, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_nn_weights('h1_x', 'decoder', [num_classes + latent_dim, hidden_dim])
        w_h2, b_h2 = create_nn_weights('h2_x', 'decoder', [hidden_dim, hidden_dim])

        w_mu, b_mu = create_nn_weights('mu_x', 'decoder', [hidden_dim, latent_dim])
        w_logvar, b_logvar = create_nn_weights('var_x', 'decoder', [hidden_dim, latent_dim])

        h1 = mlp_neuron(tf.concat([y, z], axis=1), w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)

        x_logvar = mlp_neuron(h2, w_logvar, b_logvar, activation=False)
        x_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)

        return x_mu, x_logvar
