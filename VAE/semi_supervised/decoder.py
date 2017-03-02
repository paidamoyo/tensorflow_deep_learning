import tensorflow as tf

from VAE.utils.distributions import draw_norm
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, activated_neuron, non_activated_neuron


def generator_network(FLAGS, y, z, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h3, b_h3 = create_h_weights('h3', 'decoder', [FLAGS['latent_dim'], FLAGS['decoder_h_dim']])
        w_h4, b_h4 = create_h_weights('h4', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['decoder_h_dim']])
        w_mu, b_mu = create_h_weights('mu', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['input_dim']])
        w_var, b_var = create_h_weights('var', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['input_dim']])
        # Model
        # Decoder hidden layer
        z1 = pz1_given_z2y(FLAGS=FLAGS, y=y, z2=z, reuse=True)
        h3 = activated_neuron(z1, w_h3, b_h3)
        h4 = activated_neuron(h3, w_h4, b_h4)
        # Reconstruction layer
        x_mu = non_activated_neuron(h4, w_mu, b_mu)
        x_logvar = non_activated_neuron(h4, w_var, b_var)
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
        return x_mu, x_logvar


def pz1_given_z2y(FLAGS, y, z2, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        w_h1, b_h1 = create_h_weights('h1_z', 'decoder',
                                      [FLAGS['latent_dim'] + FLAGS['num_classes'], FLAGS['decoder_h_dim']])

        w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1_decoder',
                                                                [FLAGS['decoder_h_dim'], FLAGS['latent_dim']])
        # Model
        # Decoder hidden layer
        h1 = activated_neuron(tf.concat([y, z2], axis=1), w_h1, b_h1)
        print("h1 decoder:{}, ".format(h1))

        # Z1 latent layer mu and var
        logvar_z1 = non_activated_neuron(h1, w_var_z1, b_var_z1)
        mu_z1 = non_activated_neuron(h1, w_mu_z1, b_mu_z1)
        return draw_norm(FLAGS['latent_dim'], mu_z1, logvar_z1)
