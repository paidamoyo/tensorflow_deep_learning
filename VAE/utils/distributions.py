import numpy as np
import tensorflow as tf

logc = np.log(2. * np.pi)
c = - 0.5 * np.log(2 * np.pi)


def tf_normal_logpdf(x, mu, log_sigma_sq):
    return - 0.5 * logc - log_sigma_sq / 2. - tf.div(tf.squared_difference(x, mu), 2 * tf.exp(log_sigma_sq))


def tf_stdnormal_logpdf(x):
    return - 0.5 * (logc + tf.square(x))


def tf_gaussian_ent(log_sigma_sq):
    return - 0.5 * (logc + 1.0 + log_sigma_sq)


def tf_gaussian_marg(mu, log_sigma_sq):
    return - 0.5 * (logc + (tf.square(mu) + tf.exp(log_sigma_sq)))


def tf_binary_xentropy(x, y, const=1e-10):
    return - (x * tf.log(tf.clip_by_value(y, const, 1.0)) + (1.0 - x) * tf.log(tf.clip_by_value(1.0 - y, const, 1.0)))


def draw_z(dim, mu, logvar):
    epsilon_encoder = tf.random_normal(tf.shape(dim), name='epsilon')
    std_encoder_z1 = tf.exp(0.5 * logvar)
    return mu + tf.multiply(std_encoder_z1, epsilon_encoder)
