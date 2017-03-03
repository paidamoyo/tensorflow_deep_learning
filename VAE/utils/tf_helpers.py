import tensorflow as tf


def mlp_neuron(layer_input, weights, biases, activation=True):
    if activation:
        return tf.nn.relu(tf.add(tf.matmul(layer_input, weights), biases))
    else:
        return tf.add(tf.matmul(layer_input, weights), biases)


def create_h_weights(layer, network, shape):
    h_vars = {}
    w_h = 'W_' + network + '_' + layer
    b_h = 'b_' + network + '_' + layer
    print("layer:{}, network:{}, shape:{}".format(layer, network, shape))
    h_vars[w_h] = create_weights(shape)
    h_vars[b_h] = create_biases([shape[1]])
    variable_summaries(h_vars[w_h], w_h)
    variable_summaries(h_vars[b_h], b_h)

    return h_vars[w_h], h_vars[b_h]


def create_z_weights(layer, shape):
    print("layer:{}, z_latent, shape:{}".format(layer, shape))
    z_vars = {}
    network = 'encoder'

    # Mean
    w_z_mu = 'W_' + network + 'mu_' + layer
    b_z_mu = 'b_' + network + 'mu_' + layer
    z_vars[w_z_mu] = create_weights(shape)
    z_vars[b_z_mu] = create_biases([shape[1]])
    variable_summaries(z_vars[w_z_mu], w_z_mu)
    variable_summaries(z_vars[b_z_mu], b_z_mu)

    # Variance
    w_z_var = 'W_' + network + 'var_' + layer
    b_z_var = 'b_' + network + 'var_' + layer
    z_vars[w_z_var] = create_weights(shape)
    z_vars[b_z_var] = create_biases([shape[1]])
    variable_summaries(z_vars[w_z_var], w_z_var)
    variable_summaries(z_vars[b_z_var], b_z_var)

    return z_vars[w_z_mu], z_vars[w_z_var], z_vars[b_z_mu], z_vars[b_z_var]


def create_biases(shape):
    return tf.Variable(tf.random_normal(shape))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def variable_summaries(var, summary_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def one_label_tensor(label, num_ulab_batch, num_classes):
    indices = []
    values = []
    for i in range(num_ulab_batch):
        indices += [[i, label]]
        values += [1.]
    y_ulab = tf.sparse_tensor_to_dense(
        tf.SparseTensor(indices=indices, values=values, dense_shape=[num_ulab_batch, num_classes]), 0.0)
    return y_ulab


if __name__ == '__main__':
    y_ulab = one_label_tensor(0, 400, 10)
    with tf.Session() as session:
        y = session.run(y_ulab)
        print("y:{}, shape:{}".format(y, y.shape))
