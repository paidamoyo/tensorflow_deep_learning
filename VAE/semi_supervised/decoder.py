import tensorflow as tf

from VAE.utils.tf_helpers import create_h_weights, activated_neuron, non_activated_neuron


def generator_network(FLAGS, y, z, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h3, b_h3 = create_h_weights('h3', 'decoder',
                                      [FLAGS['latent_dim'] + FLAGS['num_classes'], FLAGS['decoder_h_dim']])
        w_mu, b_mu = create_h_weights('mu', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['input_dim']])
        w_var, b_var = create_h_weights('var', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['input_dim']])

        # Decoder hidden layer
        h3 = activated_neuron(tf.concat((y, z), axis=1), w_h3, b_h3)

        # Reconstruction layer
        x_mu = non_activated_neuron(h3, w_mu, b_mu)
        x_logvar = non_activated_neuron(h3, w_var, b_var)
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
        return x_mu, x_logvar
