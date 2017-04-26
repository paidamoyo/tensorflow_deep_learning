import tensorflow as tf

from models.utils.distributions import draw_norm
from models.utils.tf_helpers import conv_layer, fc_layer, flatten_layer


def q_z1_given_x(x, num_channels, filter_sizes, num_filters, latent_dim, fc_size, reuse=False):
    with tf.variable_scope("encoder_M1", reuse=reuse):
        layer_conv1, weights_conv1 = conv_layer(input=x, num_input_channels=num_channels, filter_size=filter_sizes[0],
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

        # ### Fully-Connected Layer 2
        logvar_z1 = fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=latent_dim, use_relu=False)
        mu_z1 = fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=latent_dim, use_relu=False)

        # Model
        z1 = draw_norm(latent_dim, mu_z1, logvar_z1)
        return z1, mu_z1, logvar_z1
