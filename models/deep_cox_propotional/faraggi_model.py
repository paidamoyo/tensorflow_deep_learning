import tensorflow as tf

from models.utils.tf_helpers import create_nn_weights, mlp_neuron, normalized_mlp


def log_hazard_ratio(x, input_dim, hidden_dim, batch_norm, is_training, reuse=False):
    # Combine Linear to output Log Hazard Ratio - same as Faraggi
    with tf.variable_scope("risk", reuse=reuse):
        # Variables
        w_h1, b_h1 = create_nn_weights('h1', 'risk', [input_dim, hidden_dim])
        w_h2, b_h2 = create_nn_weights('h2', 'risk', [hidden_dim, hidden_dim])

        w_risk, b_risk = create_nn_weights('risk', 'risk', [hidden_dim, 1])
        # Model
        # hidden layer
        h1 = normalized_mlp(x, w_h1, b_h1, is_training, batch_norm=batch_norm)
        h2 = normalized_mlp(h1, w_h2, b_h2, is_training, batch_norm=batch_norm)

        # Output layer
        fully_connected = mlp_neuron(h2, w_risk, b_risk, activation=False)

        return fully_connected
