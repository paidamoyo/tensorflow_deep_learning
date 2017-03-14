import tensorflow as tf


def mlp_neuron(layer_input, weights, biases, activation=True):
    if activation:
        return tf.nn.relu(tf.add(tf.matmul(layer_input, weights), biases))
    else:
        return tf.add(tf.matmul(layer_input, weights), biases)


def create_nn_weights(layer, network, shape):
    h_vars = {}
    w_h = 'W_' + network + '_' + layer
    b_h = 'b_' + network + '_' + layer
    h_vars[w_h] = create_weights(shape=shape, name=w_h)
    h_vars[b_h] = create_biases([shape[1]], b_h)
    variable_summaries(h_vars[w_h], w_h)
    variable_summaries(h_vars[b_h], b_h)

    return h_vars[w_h], h_vars[b_h]


def create_biases(shape, name):
    print("name:{}, shape{}".format(name, shape))
    return tf.Variable(tf.constant(shape=shape, value=0.0), name=name)


def create_weights(shape, name):
    print("name:{}, shape{}".format(name, shape))
    # initialize weights using Glorot and Bengio(2010) scheme
    a = tf.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.random_uniform(shape, minval=-a, maxval=a, dtype=tf.float32))


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


def get_variables(name):
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)[0]
    return var


def one_label_tensor(label, num_ulab_batch, num_classes):
    indices = []
    values = []
    for i in range(num_ulab_batch):
        indices += [[i, label]]
        values += [1.]
    lab = tf.sparse_tensor_to_dense(
        tf.SparseTensor(indices=indices, values=values, dense_shape=[num_ulab_batch, num_classes]), 0.0)
    return lab


if __name__ == '__main__':
    y_ulab = one_label_tensor(2, 400, 10)
    with tf.Session() as session:
        y = session.run(y_ulab)
        print("y:{}, shape:{}".format(y, y.shape))
