import tensorflow as tf

from models.utils.distributions import draw_norm
from models.utils.tf_helpers import create_h_weights, create_z_weights, mlp_neuron


def q_z_given_xy(x, y, latent_dim, num_classes, hidden_dim, reuse=False):
    with tf.variable_scope("encoder_z2", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_z2', 'encoder',
                                      [latent_dim + num_classes, hidden_dim])

        w_mu_z2, w_var_z2, b_mu_z2, b_var_z2 = create_z_weights('z_2', [hidden_dim, latent_dim])

        # Hidden layers
        h1 = mlp_neuron(tf.concat([x, y], axis=1), w_h1, b_h1)
        # Z2 latent layer mu and var
        logvar_z2 = mlp_neuron(h1, w_var_z2, b_var_z2, activation=False)
        mu_z2 = mlp_neuron(h1, w_mu_z2, b_mu_z2, activation=False)
        z2 = draw_norm(latent_dim, mu_z2, logvar_z2)
        return z2, mu_z2, logvar_z2


def qy_given_x(x, latent_dim, hidden_dim, num_classes, reuse=False):
    with tf.variable_scope("y_classifier", reuse=reuse):
        w_mlp_h1, b_mlp_h1 = create_h_weights('y_h1', 'infer', [latent_dim, hidden_dim])
        w_mlp_h2, b_mlp_h2 = create_h_weights('y_h2', 'infer', [hidden_dim, num_classes])
        h1 = mlp_neuron(x, w_mlp_h1, b_mlp_h1)
    return mlp_neuron(h1, w_mlp_h2, b_mlp_h2, activation=False)
