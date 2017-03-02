import tensorflow as tf

from VAE.utils.distributions import draw_norm
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, mlp_neuron


def px_given_z1(FLAGS, y, z, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_h_weights('h1_x', 'decoder', [FLAGS['latent_dim'], FLAGS['m1_h_dim']])
        w_h2, b_h2 = create_h_weights('h2_x', 'decoder', [FLAGS['m1_h_dim'], FLAGS['m1_h_dim']])

        w_mu, b_mu = create_h_weights('mu', 'decoder', [FLAGS['m1_h_dim'], FLAGS['input_dim']])
        w_var, b_var = create_h_weights('var', 'decoder', [FLAGS['m1_h_dim'], FLAGS['input_dim']])
        # Model
        # Decoder hidden layer
        z1 = pz1_given_z2y(FLAGS=FLAGS, y=y, z2=z, reuse=True)
        h1 = mlp_neuron(z1, w_h1, b_h1)
        h2 = mlp_neuron(h1, w_h2, b_h2)
        # Reconstruction layer
        x_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)
        x_logvar = mlp_neuron(h2, w_var, b_var, activation=False)
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
        return x_mu, x_logvar


def pz1_given_z2y(FLAGS, y, z2, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        w_h1, b_h1 = create_h_weights('h1_z', 'decoder',
                                      [FLAGS['latent_dim'] + FLAGS['num_classes'], FLAGS['m2_h_dim']])

        w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1_decoder',
                                                                [FLAGS['m2_h_dim'], FLAGS['latent_dim']])
        # Model
        # Decoder hidden layer
        h1 = mlp_neuron(tf.concat([y, z2], axis=1), w_h1, b_h1)
        print("h1 decoder:{}, ".format(h1))

        # Z1 latent layer mu and var
        logvar_z1 = mlp_neuron(h1, w_var_z1, b_var_z1, activation=False)
        mu_z1 = mlp_neuron(h1, w_mu_z1, b_mu_z1, activation=False)
        return draw_norm(FLAGS['latent_dim'], mu_z1, logvar_z1)


def reconstruction_loss(x_input, x_hat):
    return tf.reduce_sum(tf.squared_difference(x_input, x_hat), axis=1)
