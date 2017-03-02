import tensorflow as tf

from VAE.utils.distributions import draw_norm
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, mlp_neuron


def q_z1_given_x(FLAGS, x, reuse=False):
    with tf.variable_scope("encoder_z1", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1', 'M1_encoder', [FLAGS['input_dim'], FLAGS['m1_h_dim']])
        w_h2, b_h2 = create_h_weights('h2', 'M1_encoder', [FLAGS['m1_h_dim'], FLAGS['m1_h_dim']])

        w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1', [FLAGS['m1_h_dim'], FLAGS['latent_dim']])

        # Hidden layers
        h1 = mlp_neuron(x, w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)

        # Z1 latent layer mu and var
        logvar_z1 = mlp_neuron(h2, w_var_z1, b_var_z1, activation=False)
        mu_z1 = mlp_neuron(h2, w_mu_z1, b_mu_z1, activation=False)
        # Model
        z1 = draw_norm(FLAGS['latent_dim'], mu_z1, logvar_z1)
        y_logits = qy_given_x(z1, FLAGS, reuse=reuse)
        return z1, y_logits


def q_z2_given_yx(FLAGS, z1, y, reuse=False):
    with tf.variable_scope("encoder_z2", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_z', 'encoder',
                                      [FLAGS['latent_dim'] + FLAGS['num_classes'], FLAGS['m2_h_dim']])

        w_mu_z2, w_var_z2, b_mu_z2, b_var_z2 = create_z_weights('z_2', [FLAGS['m2_h_dim'], FLAGS['latent_dim']])

        # Hidden layers
        h1 = mlp_neuron(tf.concat([z1, y], axis=1), w_h1, b_h1)
        # Z2 latent layer mu and var
        logvar_z2 = mlp_neuron(h1, w_var_z2, b_var_z2, activation=False)
        mu_z2 = mlp_neuron(h1, w_mu_z2, b_mu_z2, activation=False)
        z2 = draw_norm(FLAGS['latent_dim'], mu_z2, logvar_z2)
        return z2, mu_z2, logvar_z2


def qy_given_x(z_1, FLAGS, reuse=False):
    with tf.variable_scope("y_classifier", reuse=reuse):
        num_classes = FLAGS['num_classes']
        w_mlp_h1, b_mlp_h1 = create_h_weights('y_h1', 'classifier', [FLAGS['latent_dim'], FLAGS['m2_h_dim']])
        w_mlp_h2, b_mlp_h2 = create_h_weights('y_h2', 'classifier', [FLAGS['m2_h_dim'], num_classes])
        h1 = mlp_neuron(z_1, w_mlp_h1, b_mlp_h1)
    return mlp_neuron(h1, w_mlp_h2, b_mlp_h2, activation=False)


def qz_regularization_loss(encoder_logvar_z2, encoder_mu_z2):
    z_regularization = -0.5 * tf.reduce_sum(
        1 + encoder_logvar_z2 - tf.pow(encoder_mu_z2, 2) - tf.exp(encoder_logvar_z2), axis=1)
    return z_regularization
