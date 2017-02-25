import tensorflow as tf

from VAE.utils.distributions import draw_z
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, activated_neuron, non_activated_neuron


def generator_network(FLAGS, y_logits, z_latent_rep):
    # Variables
    w_decoder_h_3, b_decoder_h_3 = create_h_weights('h3', 'decoder',
                                                    [FLAGS['latent_dim'], FLAGS['decoder_h_dim']])
    w_decoder_h_4, b_decoder_h_4 = create_h_weights('h4', 'decoder',
                                                    [FLAGS['decoder_h_dim'], FLAGS['decoder_h_dim']])
    w_decoder_mu, b_decoder_mu = create_h_weights('mu', 'decoder', [FLAGS['decoder_h_dim'], FLAGS['input_dim']])
    # Model
    # Decoder hidden layer
    decoder_h_3 = activated_neuron(pz1_given_z2y(FLAGS=FLAGS, y_logits=y_logits, z_latent_rep=z_latent_rep),
                                   w_decoder_h_3,
                                   b_decoder_h_3)
    decoder_h_4 = activated_neuron(decoder_h_3, w_decoder_h_4, b_decoder_h_4)

    # Reconstruction layer
    x_mu = non_activated_neuron(decoder_h_4, w_decoder_mu, b_decoder_mu)
    tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
    return x_mu


def pz1_given_z2y(FLAGS, y_logits, z_latent_rep):
    w_decoder_h_1_z, b_decoder_h_1_z = create_h_weights('h1_z', 'decoder',
                                                        [FLAGS['latent_dim'], FLAGS['decoder_h_dim']])
    split_dim = int(FLAGS['decoder_h_dim'] / 2)
    w_decoder_h_2_z, b_decoder_h_2_z = create_h_weights('h2_z', 'decoder',
                                                        [FLAGS['decoder_h_dim'], split_dim])

    w_decoder_h_1_y, b_decoder_h_1_y = create_h_weights('h1_y', 'decoder',
                                                        [FLAGS['num_classes'], FLAGS['decoder_h_dim']])
    w_decoder_h_2_y, b_decoder_h_2_y = create_h_weights('h2_y', 'decoder',
                                                        [FLAGS['decoder_h_dim'], split_dim])

    w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1_decoder',
                                                            [FLAGS['decoder_h_dim'], FLAGS['latent_dim']])
    # Model
    # Decoder hidden layer
    decoder_h_1_z = activated_neuron(z_latent_rep, w_decoder_h_1_z, b_decoder_h_1_z)
    decoder_h_2_z = activated_neuron(decoder_h_1_z, w_decoder_h_2_z, b_decoder_h_2_z)

    decoder_h_1_y = activated_neuron(y_logits, w_decoder_h_1_y, b_decoder_h_1_y)
    decoder_h_2_y = activated_neuron(decoder_h_1_y, w_decoder_h_2_y, b_decoder_h_2_y)

    # Z1 latent layer mu and var
    decoder_logvar_z1 = non_activated_neuron(tf.concat((decoder_h_2_y, decoder_h_2_z), axis=1), w_var_z1, b_var_z1)
    decoder_mu_z1 = non_activated_neuron(tf.concat((decoder_h_2_y, decoder_h_2_z), axis=1), w_mu_z1, b_mu_z1)
    return draw_z(FLAGS['latent_dim'], decoder_mu_z1, decoder_logvar_z1)
