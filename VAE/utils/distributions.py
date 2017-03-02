import numpy as np
import tensorflow as tf

logc = np.log(2. * np.pi)
c = - 0.5 * np.log(2 * np.pi)


def tf_normal_logpdf(x, mu, log_sigma_sq):
    return - 0.5 * logc - log_sigma_sq / 2. - tf.div(tf.squared_difference(x, mu), 2 * tf.exp(log_sigma_sq))


def tf_stdnormal_logpdf(mu):
    return - 0.5 * (logc + tf.square(mu))


def tf_gaussian_ent(log_sigma_sq):
    return - 0.5 * (logc + 1.0 + log_sigma_sq)


def tf_gaussian_marg(mu, log_sigma_sq):
    return - 0.5 * (logc + (tf.square(mu) + tf.exp(log_sigma_sq)))


def tf_binary_xentropy(x, y, const=1e-10):
    return - (x * tf.log(tf.clip_by_value(y, const, 1.0)) + (1.0 - x) * tf.log(tf.clip_by_value(1.0 - y, const, 1.0)))


def draw_norm(dim, mu, logvar):
    epsilon = tf.random_normal(tf.shape(dim), name='epsilon')
    std = tf.exp(0.5 * logvar)
    return tf.add(mu, tf.multiply(std, epsilon))


def prior_weights():
    prior_weights_loss = 0.
    weights = tf.trainable_variables()
    for w in weights:
        prior_weights_loss += tf.reduce_sum(tf_stdnormal_logpdf(w))
    tf.summary.scalar('prior_weights_loss', prior_weights_loss)
    return prior_weights_loss


def compute_ELBO(x_recon, x, y, z):
    num_classes = 10
    y_prior = (1. / num_classes) * tf.ones_like(y)
    z_prior = tf.ones_like(z[0])  # or z[0]?
    log_prior_z = tf.reduce_sum(tf_stdnormal_logpdf(mu=z_prior), axis=1)
    log_prior_y = -tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)
    log_lik = tf.reduce_sum(tf_normal_logpdf(x, x_recon[0], x_recon[1]), axis=1)
    log_post_z = tf.reduce_sum(tf_normal_logpdf(x=z[0], mu=z[1], log_sigma_sq=z[2]), axis=1)

    return log_prior_y + log_lik + log_prior_z - log_post_z
