import tensorflow as tf

from models.utils.distributions import draw_norm
from models.utils.tf_helpers import conv_layer, flatten_layer, fc_layer


def q_z2_given_z1y(z1, y, latent_dim, input_dim, fc_size, filter_sizes, num_channels,
                   num_filters, reuse=False):
    with tf.variable_scope("encoder_M2", reuse=reuse):
        z1y = tf.concat([y, z1], axis=1)
        expanded_z1y = tf.expand_dims(tf.expand_dims(z1y, 1), 1)
        print("filter_sizes:{}".format(filter_sizes))
        layer_conv1, weights_conv1 = conv_layer(input=expanded_z1y, num_input_channels=num_channels,
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

        logvar_z2 = fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=input_dim, use_relu=False)
        mu_z2 = fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=input_dim, use_relu=False)
        z2 = draw_norm(latent_dim, mu_z2, logvar_z2)

        return z2, mu_z2, logvar_z2


def qy_given_z1(z1, num_classes, filter_sizes, num_channels, num_filters, fc_size, reuse=False):
    with tf.variable_scope("y_classifier", reuse=reuse):
        print("filter_sizes:{}".format(filter_sizes))
        expanded_z1 = tf.expand_dims(tf.expand_dims(z1, 1), 1)
        layer_conv1, weights_conv1 = conv_layer(input=expanded_z1, num_input_channels=num_channels,
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

        logits = fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
    return logits
