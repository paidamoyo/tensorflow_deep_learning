import tensorflow as tf

from models.utils.distributions import draw_norm
from models.utils.tf_helpers import create_nn_weights, mlp_neuron, normalized_mlp


def px_given_zy(z, y, hidden_dim, input_dim, latent_dim, num_classes, is_training, batch_norm, reuse=False):
    # Generative p(x|z,y)
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1_z, b_h1_z = create_nn_weights('h1_x_z', 'decoder', [latent_dim, hidden_dim])
        w_h1_y, b_h1_y = create_nn_weights('h1_x_y', 'decoder', [num_classes, hidden_dim])

        w_h1, b_h1 = create_nn_weights('h1_x', 'decoder', [hidden_dim, hidden_dim])
        # w_h1, b_h1 = create_nn_weights('h1_x', 'decoder', [latent_dim + num_classes, hidden_dim])
        w_h2, b_h2 = create_nn_weights('h2_x', 'decoder', [hidden_dim, hidden_dim])

        w_mu, b_mu = create_nn_weights('mu_x', 'decoder', [hidden_dim, input_dim])
        # Model
        # Decoder hidden layer
        l_y_to_px = mlp_neuron(y, w_h1_y, b_h1_y, activation=False)
        l_qz_to_px = mlp_neuron(z, w_h1_z, b_h1_z, activation=False)

        h1 = normalized_mlp(tf.add(l_y_to_px, l_qz_to_px), w_h1, b_h1, is_training, batch_norm=batch_norm)
        h2 = normalized_mlp(h1, w_h2, b_h2, is_training, batch_norm=batch_norm)

        # Reconstruction layer
        # x_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)
        fully_connected = mlp_neuron(h2, w_mu, b_mu, activation=False)  # TODO look at activation?

        x_mu = tf.nn.sigmoid(fully_connected)
        return x_mu


def pa_given_zy(z, y, hidden_dim, latent_dim, num_classes, is_training, batch_norm, reuse=False):
    # Generative p(a|z,y)
    with tf.variable_scope("decoder", reuse=reuse):
        # Variables
        w_h1_z, b_h1_z = create_nn_weights('h1_a_z', 'decoder', [latent_dim, hidden_dim])
        w_h1_y, b_h1_y = create_nn_weights('h1_a_y', 'decoder', [num_classes, hidden_dim])

        w_h1, b_h1 = create_nn_weights('h1_a', 'decoder', [hidden_dim, hidden_dim])
        # w_h1, b_h1 = create_nn_weights('h1_a', 'decoder', [latent_dim + num_classes, hidden_dim])
        w_h2, b_h2 = create_nn_weights('h2_a', 'decoder', [hidden_dim, hidden_dim])

        w_mu_a, b_mu_a = create_nn_weights('mu_a', 'decoder', [hidden_dim, latent_dim])
        w_var_a, b_var_a = create_nn_weights('var_a', 'encoder', [hidden_dim, latent_dim])
        # Model
        # Decoder hidden layer
        l_y_to_pa = mlp_neuron(y, w_h1_y, b_h1_y, activation=False)
        l_qz_to_pa = mlp_neuron(z, w_h1_z, b_h1_z, activation=False)
        h1 = normalized_mlp(tf.add(l_y_to_pa, l_qz_to_pa), w_h1, b_h1, is_training, batch_norm=batch_norm)
        h2 = normalized_mlp(h1, w_h2, b_h2, is_training, batch_norm=batch_norm)

        # a latent layer mu and var
        logvar_a = mlp_neuron(h2, w_var_a, b_var_a, activation=False)
        mu_a = mlp_neuron(h2, w_mu_a, b_mu_a, activation=False)
        # Model
        a = draw_norm(latent_dim, mu_a, logvar_a)
        return a, mu_a, logvar_a
