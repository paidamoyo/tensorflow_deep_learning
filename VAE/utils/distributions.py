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


# https://github.com/saemundsson/semisupervised_vae/blob/master/vae.py

def compute_ELBO(x_recon, x, y, z):
    # if self.distributions['p_z'] == 'gaussian_marg'
    # log_prior_z = tf.reduce_sum(tf_gaussian_marg(z[1], z[2]), 1)

    # elif self.distributions['p_z'] == 'gaussian':
    log_prior_z = tf.reduce_sum(tf_stdnormal_logpdf(z[0]), 1)

    # if self.distributions['p_y'] == 'uniform':

    num_classes = 10
    y_prior = (1. / num_classes) * tf.ones_like(y)
    log_prior_y = - tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)

    # if self.distributions['p_x'] == 'gaussian':
    log_lik = tf.reduce_sum(tf_normal_logpdf(x, x_recon[0], x_recon[1]), 1)

    # if self.distributions['q_z'] == 'gaussian_marg':
    # log_post_z = tf.reduce_sum(tf_gaussian_ent(z[2]), 1)

    # elif self.distributions['q_z'] == 'gaussian':
    log_post_z = tf.reduce_sum(tf_normal_logpdf(z[0], z[1], z[2]), 1)
    print("log_prior_y:{}, log_lik:{}, log_prior_z:{}, log_post_z:{}".format(log_prior_y.shape, log_lik.shape,
                                                                             log_prior_z.shape, log_post_z.shape))
    return log_prior_y + log_lik + log_prior_z - log_post_z
