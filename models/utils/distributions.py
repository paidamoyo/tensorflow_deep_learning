import numpy as np
import tensorflow as tf

logc = np.log(2. * np.pi)
c = - 0.5 * np.log(2 * np.pi)


def tf_normal_logpdf(x, mu, log_var):
    return - 0.5 * logc - log_var / 2. - tf.div(tf.squared_difference(x, mu), 2 * tf.exp(log_var))


def tf_stdnormal_logpdf(x):
    return - 0.5 * (logc + tf.square(x))


def tf_gaussian_ent(log_sigma_sq):
    return - 0.5 * (logc + 1.0 + log_sigma_sq)


def tf_gaussian_marg(mu, log_sigma_sq):
    return - 0.5 * (logc + (tf.square(mu) + tf.exp(log_sigma_sq)))


def tf_binary_xentropy(x_true, x_approx, const=1e-10):
    print("x_approx:{}, x_true:{}".format(x_approx, x_true))
    return - (x_true * tf.log(tf.clip_by_value(x_approx, const, 1.0)) + tf.subtract(1.0, x_true) * tf.log(
        tf.clip_by_value(tf.subtract(1.0, x_approx), const, 1.0)))


def l2_loss():
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return l2


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

    log_lik = tf.reduce_sum(tf_normal_logpdf(x=z1, mu=z1_recon[0], log_var=z1_recon[1]), 1)
    # log_post_z = tf.reduce_sum(tf_gaussian_ent(z2[2]), 1)
    # log_prior_z = tf.reduce_sum(tf_gaussian_marg(z2[1], z2[2]), 1)
    marg_prior_z = tf_gaussian_marg(tf.zeros_like(z2[1]), tf.ones_like(z2[2]))
    marg_post_z = tf_gaussian_marg(z2[1], z2[2])
    log_prior_z = tf.reduce_sum(marg_prior_z, axis=1)
    log_post_z = tf.reduce_sum(marg_post_z, axis=1)
    return log_prior_y + log_lik + log_prior_z - log_post_z


def elbo_M1(x_recon, x_true, z1, z1_mu, z1_lsgms):
    log_lik = -tf.reduce_sum(tf_binary_xentropy(x_true=x_true, x_approx=x_recon))
    # log_post_z = tf.reduce_sum(tf_gaussian_ent(z1_lsgms), axis=1)
    # log_prior_z = tf.reduce_sum(tf_gaussian_marg(z1_mu, z1_lsgms), axis=1)
    marg_prior_z = tf_gaussian_marg(tf.zeros_like(z1_mu), tf.ones_like(z1_lsgms))
    marg_post_z = tf_gaussian_marg(z1_mu, z1_lsgms)
    marginal_lik = tf.reduce_sum((marg_prior_z * log_lik) / marg_post_z, axis=1)

    log_prior_z = tf.reduce_sum(marg_prior_z, axis=1)
    log_post_z = tf.reduce_sum(marg_post_z, axis=1)
    negative_log_lik = tf.scalar_mul(-1, log_lik)
    tf.summary.scalar('negative_log_lik', negative_log_lik)
    cost = log_lik + log_prior_z - log_post_z
    print("M1 cost {}, {}, {}".format(log_lik, log_post_z, log_prior_z))
    print("z1 shape:{}".format(z1.shape))
    return cost, marginal_lik


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


def auxiliary_elbo(x_recon, x, y, qz, qa, pa):
    num_classes = 10
    y_prior = (1. / num_classes) * tf.ones_like(y)

    print(x, x_recon)
    log_px = -tf.reduce_sum(tf_binary_xentropy(x_true=x, x_approx=x_recon))
    logpdf_post_z = tf_normal_logpdf(x=qz[0], mu=qz[1], log_var=qz[2])
    log_qz = tf.reduce_sum(logpdf_post_z, 1)
    log_qa = tf.reduce_sum(tf_normal_logpdf(x=qa[0], mu=qa[1], log_var=qa[2]))

    logpdf_prior_z = tf_stdnormal_logpdf(x=qa[0])
    log_pz = tf.reduce_sum(logpdf_prior_z)
    log_py = -tf.nn.softmax_cross_entropy_with_logits(logits=y_prior, labels=y)
    log_pa = tf.reduce_sum(tf_normal_logpdf(x=qa[0], mu=pa[1], log_var=pa[2]))

    negative_log_lik = tf.scalar_mul(-1, log_px)
    tf.summary.scalar('negative_log_lik', negative_log_lik)
    elbo = log_px + log_py + log_pz + log_pa - log_qa - log_qz
    marginal_lik = tf.reduce_sum((logpdf_prior_z * log_px) / logpdf_post_z)
    return elbo, marginal_lik
