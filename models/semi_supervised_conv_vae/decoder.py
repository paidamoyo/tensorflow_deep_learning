import tensorflow as tf

from  models.utils.distributions import draw_norm
from models.utils.tf_helpers import create_nn_weights, mlp_neuron


def pz1_given_z2y(y, z2, latent_dim, num_classes, hidden_dim, input_dim, reuse=False):
    with tf.variable_scope("decoder_M2", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_nn_weights('h1_z1', 'decoder', [num_classes + latent_dim, hidden_dim])
        w_h2, b_h2 = create_nn_weights('h2_z1', 'decoder', [hidden_dim, hidden_dim])

        w_mu, b_mu = create_nn_weights('mu_z1', 'decoder', [hidden_dim, input_dim])
        w_logvar, b_logvar = create_nn_weights('var_z1', 'decoder', [hidden_dim, input_dim])

        h1 = mlp_neuron(tf.concat([y, z2], axis=1), w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)

        z1_logvar = mlp_neuron(h2, w_logvar, b_logvar, activation=False)
        z1_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)
        z1 = draw_norm(input_dim, z1_mu, z1_logvar)

        return z1, z1_mu, z1_logvar
