import tensorflow as tf

from VAE.utils.distributions import draw_norm
from VAE.utils.tf_helpers import create_h_weights, create_z_weights, activated_neuron, non_activated_neuron


def q_z_1_given_x(FLAGS, x):
    # Variables
    w_h1, b_h1 = create_h_weights('h1', 'encoder', [FLAGS['input_dim'], FLAGS['encoder_h_dim']])
    w_h2, b_h2 = create_h_weights('h2', 'encoder',
                                  [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])
    w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = create_z_weights('z_1', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Hidden layers
    h1 = activated_neuron(x, w_h1, b_h1)
    h2 = activated_neuron(h1, w_h2, b_h2)

    # Z1 latent layer mu and var
    logvar_z1 = non_activated_neuron(h2, w_var_z1, b_var_z1)
    mu_z1 = non_activated_neuron(h2, w_mu_z1, b_mu_z1)
    return draw_norm(FLAGS['latent_dim'], mu_z1, logvar_z1)


def recognition_network(FLAGS, x):
    # Variables
    w_h3_z1, b_h3_z1 = create_h_weights('h3_z1', 'encoder', [FLAGS['latent_dim'], FLAGS['encoder_h_dim']])
    w_h4_z1, b_h4_z1 = create_h_weights('h4_z1', 'encoder', [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])

    w_h3_y, b_h3_y = create_h_weights('h3_y', 'encoder', [FLAGS['num_classes'], FLAGS['encoder_h_dim']])
    w_h4_y, b_h4_y = create_h_weights('h4_y', 'encoder', [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])

    w_h4_mu, b_h4_mu = create_h_weights('h4_mu', 'encoder', [2 * FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])

    w_mu_z2, w_var_z2, b_mu_z2, b_var_z2 = create_z_weights('z_2', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Model
    z_1 = q_z_1_given_x(FLAGS, x)
    y_logits, weights, y_regularization = qy_given_x(z_1, FLAGS)

    # Hidden layers
    h3_z1 = activated_neuron(z_1, w_h3_z1, b_h3_z1)
    h4_z1 = activated_neuron(h3_z1, w_h4_z1, b_h4_z1)

    h3_y = activated_neuron(y_logits, w_h3_y, b_h3_y)
    h4_y = activated_neuron(h3_y, w_h4_y, b_h4_y)

    h4_mu = activated_neuron(tf.concat((h4_y, h4_z1), axis=1), w_h4_mu, b_h4_mu)

    # Z2 latent layer mu and var
    logvar_z2 = non_activated_neuron(h4_z1, w_var_z2, b_var_z2)
    mu_z2 = non_activated_neuron(h4_mu, w_mu_z2, b_mu_z2)
    return mu_z2, logvar_z2, y_logits


def qy_given_x(z_1, FLAGS):
    num_classes = FLAGS['num_classes']
    w_mlp_h1, b_mlp_h1 = create_h_weights('y_h1', 'classifier', [FLAGS['latent_dim'], num_classes])

    logits = non_activated_neuron(z_1, w_mlp_h1, b_mlp_h1)
    pi = 1 / num_classes
    y_prior = tf.constant(value=pi, shape=[num_classes])
    print("logits:{}, y_prior:{}, reduced_sum:{}".format(logits.shape, y_prior, tf.reduce_sum(y_prior, axis=0)))
    mean_class_logits = tf.reduce_mean(logits, axis=0)
    regularization = tf.nn.softmax_cross_entropy_with_logits(labels=y_prior, logits=mean_class_logits)
    print("regularization:{}".format(regularization))
    return logits, w_mlp_h1, regularization


def qz_regularization_loss(encoder_logvar_z2, encoder_mu_z2):
    z_regularization = -0.5 * tf.reduce_sum(
        1 + encoder_logvar_z2 - tf.pow(encoder_mu_z2, 2) - tf.exp(encoder_logvar_z2), axis=1)
    return z_regularization
