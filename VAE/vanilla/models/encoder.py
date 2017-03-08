import tensorflow as tf

from VAE.utils.distributions import draw_norm
from VAE.utils.settings import initialize
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, mlp_neuron

FLAGS = initialize()


def q_z1_given_x(x, reuse=False):
    with tf.variable_scope("encoder_z1", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_z1', 'M1_encoder', [FLAGS['input_dim'], FLAGS['m1_h_dim']])
        w_h2, b_h2 = create_h_weights('h2_z1', 'M1_encoder', [FLAGS['m1_h_dim'], FLAGS['m1_h_dim']])

        w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1', [FLAGS['m1_h_dim'], FLAGS['latent_dim']])

        # Hidden layers
        h1 = mlp_neuron(x, w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)

        # Z1 latent layer mu and var
        logvar_z1 = mlp_neuron(h2, w_var_z1, b_var_z1, activation=False)
        mu_z1 = mlp_neuron(h2, w_mu_z1, b_mu_z1, activation=False)
        # Model
        z1 = draw_norm(FLAGS['latent_dim'], mu_z1, logvar_z1)
        return z1, mu_z1, logvar_z1
