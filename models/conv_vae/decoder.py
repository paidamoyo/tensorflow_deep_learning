import tensorflow as tf

from models.utils.tf_helpers import conv_layer, fc_layer, flatten_layer


def px_given_z1(z1, input_dim, num_channels, filter_sizes, fc_size, num_filters, reuse=False):
    with tf.variable_scope("decoder_M1", reuse=reuse):
        # Variables
        print("filter_sizes:{}".format(filter_sizes))
        layer_conv1, weights_conv1 = conv_layer(input=z1, num_input_channels=num_channels,
                                                filter_size=filter_sizes[0],
                                                num_filters=num_filters[0], use_pooling=True, layer_name='layer1')
        print("layer conv1: {}".format(layer_conv1))

        # ### Convolutional Layer 2
        layer_conv2, weights_conv2 = conv_layer(input=layer_conv1, num_input_channels=num_filters[0],
                                                filter_size=filter_sizes[1], num_filters=num_filters[1],
                                                use_pooling=True, layer_name='layer2')
        print("layer conv2: {}".format(layer_conv2))

        # ### Flatten Layer
        layer_flat, num_features = flatten_layer(layer_conv2)
        print("layer flat: {}".format(layer_flat))
        print("num_features: {}".format(num_features))

        # ### Fully-Connected Layer 1
        layer_fc1 = fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
        print("layer fc1: {}".format(layer_fc1))

        # x_mu = mlp_neuron(h2, w_mu, b_mu, activation=False)
        x_mu = tf.nn.sigmoid(fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=input_dim, use_relu=False))
        tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))

    return x_mu
