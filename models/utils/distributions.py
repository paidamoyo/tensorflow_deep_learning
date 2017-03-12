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


def tf_binary_xentropy(x_true, x_approx, const=1e-10):
    print("x_approx:{}, x_true:{}".format(x_approx, x_true))
    return - (x_true * tf.log(tf.clip_by_value(x_approx, const, 1.0)) + tf.subtract(1.0, x_true) * tf.log(
        tf.clip_by_value(tf.subtract(1.0, x_approx), const, 1.0)))


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


def reconstruction_loss(x_input, x_hat):
    return tf.reduce_sum(tf.squared_difference(x_input, x_hat), axis=1)


def regularization_loss(z_mu, z_logvar):
    z_regularization = -0.5 * tf.reduce_sum(1 + z_logvar - tf.pow(z_mu, 2) - tf.exp(z_logvar), axis=1)
    return z_regularization


def elbo_M2(z1_recon, z1, y, z2):
    num_classes = 10
    y_prior = (1. / num_classes) * tf.ones_like(y)
    log_prior_y = - tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)

    log_lik = tf.reduce_sum(tf_normal_logpdf(x=z1, mu=z1_recon[0], log_sigma_sq=z1_recon[1]), 1)
    log_post_z = tf.reduce_sum(tf_gaussian_ent(z2[2]), 1)
    log_prior_z = tf.reduce_sum(tf_gaussian_marg(z2[1], z2[2]), 1)

    return log_prior_y + log_lik + log_prior_z - log_post_z


def elbo_M1(x_recon, x_true, z1, z1_mu, z1_lsgms):
    log_lik = -tf.reduce_sum(tf_binary_xentropy(x_true=x_true, x_approx=x_recon))
    log_post_z = tf.reduce_sum(tf_gaussian_ent(z1_lsgms), axis=1)
    log_prior_z = tf.reduce_sum(tf_gaussian_marg(z1_mu, z1_lsgms), axis=1)

    negative_log_lik = tf.scalar_mul(-1, log_lik)
    tf.summary.scalar('negative_log_lik', negative_log_lik)
    cost = log_lik + log_prior_z - log_post_z
    print("M1 cost {}, {}, {}".format(log_lik, log_post_z, log_prior_z))
    return cost, log_lik


def elbo_M1_M2(x_recon, z1_recon, xtrue, y, z2, z1):
    m1_cost = elbo_M1(x_recon=x_recon, x_true=xtrue, z1=z1[0], z1_mu=z1[1], z1_lsgms=z1[2])[0]
    m2_cost = elbo_M2(z1_recon=z1_recon, z1=z1[0], y=y, z2=z2)
    print("costs {} {}".format(m1_cost.shape, m2_cost.shape))
    cost = tf.add(m1_cost, m2_cost)
    return cost


def compute_ELBO(x_recon, x, y, z):
    num_classes = 10
    y_prior = (1. / num_classes) * tf.ones_like(y)

    log_prior_z = tf.reduce_sum(tf_gaussian_marg(z[1], z[2]), 1)
    log_prior_y = -tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)
    log_lik = -tf.reduce_sum(tf_binary_xentropy(x_true=x, x_approx=x_recon))
    log_post_z = tf.reduce_sum(tf_gaussian_ent(z[2]), 1)
    negative_log_lik = tf.scalar_mul(-1, log_lik)
    tf.summary.scalar('negative_log_lik', negative_log_lik)
    # log_prior_y - tf.add(reconstruction_loss(x, x_recon[0]), regularization_loss(z[1], z[2]))
    return log_prior_y + log_lik + log_prior_z - log_post_z
