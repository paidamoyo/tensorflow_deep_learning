import tensorflow as tf

from VAE.utils.distributions import draw_z
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, activated_neuron, non_activated_neuron


def q_z_1_given_x(FLAGS, x):
    # Variables
    w_encoder_h_1, b_encoder_h_1 = create_h_weights('h1', 'encoder', [FLAGS['input_dim'], FLAGS['encoder_h_dim']])
    w_encoder_h_2, b_encoder_h_2 = create_h_weights('h2', 'encoder',
                                                    [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])
    w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Hidden layers
    encoder_h_1 = activated_neuron(x, w_encoder_h_1, b_encoder_h_1)
    encoder_h_2 = activated_neuron(encoder_h_1, w_encoder_h_2, b_encoder_h_2)

    # Z1 latent layer mu and var
    encoder_logvar_z1 = non_activated_neuron(encoder_h_2, w_var_z1, b_var_z1)
    encoder_mu_z1 = non_activated_neuron(encoder_h_2, w_mu_z1, b_mu_z1)
    return draw_z(FLAGS['latent_dim'], encoder_mu_z1, encoder_logvar_z1)


def recognition_network(FLAGS, x):
    # Variables
    w_encoder_h_3, b_encoder_h_3 = create_h_weights('h3', 'encoder',
                                                    [FLAGS['latent_dim'], FLAGS['encoder_h_dim']])
    w_encoder_h_4, b_encoder_h_4 = create_h_weights('h4', 'encoder',
                                                    [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])
    w_encoder_h_4_mu, b_encoder_h_4_mu = create_h_weights('h4_mu', 'encoder', [FLAGS['encoder_h_dim'],
                                                                               FLAGS[
                                                                                   'encoder_h_dim'] - FLAGS[
                                                                                   'num_classes']])

    w_mu_z2, w_var_z2, b_mu_z2, b_var_z2 = create_z_weights('z_2', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Model
    z_1 = q_z_1_given_x(FLAGS, x)
    # Hidden layers
    encoder_h_3 = activated_neuron(z_1, w_encoder_h_3, b_encoder_h_3)
    encoder_h_4 = activated_neuron(encoder_h_3, w_encoder_h_4, b_encoder_h_4)
    encoder_h_4_mu = activated_neuron(encoder_h_3, w_encoder_h_4_mu, b_encoder_h_4_mu)

    # Z2 latent layer mu and var
    y_logits = qy_given_x(z_1, FLAGS)
    encoder_logvar_z2 = non_activated_neuron(encoder_h_4, w_var_z2, b_var_z2)
    encoder_mu_z2 = non_activated_neuron(tf.concat((y_logits, encoder_h_4_mu), axis=1), w_mu_z2,
                                         b_mu_z2)
    z_2 = draw_z(FLAGS['latent_dim'], encoder_mu_z2, encoder_logvar_z2)

    # regularization loss
    regularization = calculate_regularization_loss(encoder_logvar_z2, encoder_mu_z2)

    return z_2, regularization, y_logits


def qy_given_x(z_1, FLAGS):
    w_mlp_h1, b_mlp_h1 = create_h_weights('mlp_h1', 'classifier', [FLAGS['latent_dim'], FLAGS['latent_dim']])
    w_mlp_h2, b_mlp_h2 = create_h_weights('mlp_h2', 'classifier', [FLAGS['latent_dim'], FLAGS['num_classes']])

    h1 = activated_neuron(z_1, w_mlp_h1, b_mlp_h1)
    return non_activated_neuron(h1, w_mlp_h2, b_mlp_h2)


def calculate_regularization_loss(encoder_logvar_z2, encoder_mu_z2):
    return -0.5 * tf.reduce_sum(1 + encoder_logvar_z2 - tf.pow(encoder_mu_z2, 2) - tf.exp(encoder_logvar_z2),
                                axis=1)
