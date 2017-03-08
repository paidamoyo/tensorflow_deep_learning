import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from VAE.classifier import softmax_classifier
from VAE.semi_supervised.models.decoder import px_given_z1
from VAE.semi_supervised.models.encoder import q_z2_given_yx, qy_given_x
from VAE.utils.MNIST_pickled_preprocess import extract_data
from VAE.utils.batch_processing import get_batch_size, get_next_batch
from VAE.utils.distributions import draw_norm
from VAE.utils.distributions import elbo_M2
from VAE.utils.distributions import prior_weights
from VAE.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images
from VAE.utils.tf_helpers import one_label_tensor, variable_summaries

sys.path.append(os.getcwd())


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    best_validation_accuracy = 0
    last_improvement = 0

    start_time = time.time()
    idx_labeled = 0
    idx_unlabeled = 0

    for i in range(num_iterations):

        # Batch Training
        x_l_mu, x_l_logvar, y_l_batch, idx_labeled = get_next_batch(train_x_l_mu, train_x_l_logvar, train_l_y,
                                                                    idx_labeled,
                                                                    num_lab_batch)
        x_u_mu, x_u_logvar, _, idx_unlabeled = get_next_batch(train_x_u_mu, train_x_u_logvar, train_u_y, idx_unlabeled,
                                                              num_ulab_batch)
        feed_dict_train = {x_lab_mu: x_l_mu, y_lab: y_l_batch, x_unlab_mu: x_u_mu, x_lab_logvar: x_l_logvar,
                           x_unlab_logvar: x_u_logvar}
        summary, batch_loss, _ = session.run([merged, cost, optimizer], feed_dict=feed_dict_train)
        # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
        train_writer.add_summary(summary, i)

        if (i % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy
            correct, _ = predict_cls(images=valid_x,
                                     labels=valid_y,
                                     cls_true=convert_labels_to_cls(valid_y))
            acc_validation, _ = cls_accuracy(correct)
            if acc_validation > best_validation_accuracy:
                # Save  Best Perfoming all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=FLAGS['save_path'])
                # update best validation accuracy
                best_validation_accuracy = acc_validation
                last_improvement = i
                improved_str = '*'
            else:
                improved_str = ''

            print("Optimization Iteration: {}, Training Loss: {}, "
                  " Validation Acc:{}, {}".format(i + 1, batch_loss, acc_validation, improved_str))
        if i - last_improvement > FLAGS['require_improvement']:
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
    # Ending time.
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def reconstruct(x_test_mu, x_test_logvar, y_test):
    return session.run(x_recon_lab_mu, feed_dict={x_lab_mu: x_test_mu, x_lab_logvar: x_test_logvar, y_lab: y_test})


def test_reconstruction():
    num_images = 20
    x_test = test_x[0:num_images, ]
    y_test = test_y[0:num_images, ]
    plot_images(x_test, reconstruct(x_test, y_test), num_images)


def total_lab_loss():
    global y_pred_cls
    # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
    beta = FLAGS['alpha'] * (float(batch_size) / num_lab_batch)
    classifier_loss, y_pred_cls = softmax_classifier(logits=y_lab_logits, y_true=y_lab)
    weighted_classifier_loss = beta * classifier_loss
    labeled_loss = tf.reduce_sum(tf.subtract(labeled_ELBO, weighted_classifier_loss))
    tf.summary.scalar('labeled_loss', labeled_loss)
    return labeled_loss


def total_unlab_loss():
    # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
    const = 1e-10
    y_ulab = tf.nn.softmax(logits=y_ulab_logits)
    variable_summaries(y_lab, 'y_lab')
    weighted_elbo = tf.reduce_sum(tf.multiply(y_ulab + const, tf.subtract(unlabeled_ELBO, tf.log(y_lab + const))), 1)
    unlabeled_loss = tf.reduce_sum(weighted_elbo)
    print("unlabeled_ELBO:{}, unlabeled_loss:{}".format(unlabeled_ELBO, unlabeled_loss))
    tf.summary.scalar('unlabeled_loss', unlabeled_loss)
    return unlabeled_loss


def predict_cls(mu, logvar, labels, cls_true):
    num_images = len(mu)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)
        batch_mu = mu[i:j, :]
        batch_logavar = logvar[i:j, :]
        batch_labels = labels[i:j, :]
        feed_dict = {x_lab_mu: batch_mu,
                     x_lab_logvar: batch_logavar,
                     y_lab: batch_labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred


def train_test():
    train_neural_network(FLAGS['num_iterations'])
    saver.restore(sess=session, save_path=FLAGS['save_path'])
    correct, cls_pred = predict_cls(images=test_x,
                                    labels=test_y,
                                    cls_true=(convert_labels_to_cls(test_y)))
    print_test_accuracy(correct, cls_pred, test_y)
    test_reconstruction()


def unlabeled_model():
    # Ulabeled
    x_unlab = draw_norm(dim=FLAGS['latent_dim'], mu=x_unlab_mu, logvar=x_unlab_logvar)
    logits = qy_given_x(x_unlab, FLAGS, reuse=True)
    for label in range(FLAGS['num_classes']):
        y_ulab = one_label_tensor(label, num_ulab_batch, FLAGS['num_classes'])
        z2, z2_mu, z2_logvar = q_z2_given_yx(FLAGS, x_unlab, y_ulab, reuse=True)
        x_mu = px_given_z1(FLAGS=FLAGS, y=y_ulab, z=z2, reuse=True)
        _elbo = tf.expand_dims(elbo_M2(x_recon=x_mu, x=x_unlab, y=y_ulab, z=[z2, z2_mu, z2_logvar]), 1)
        class_elbo = tf.identity(_elbo)
        if label > 0:
            class_elbo = tf.concat((class_elbo, _elbo), axis=1)  # Decoder Model
        print("unlabeled class_elbo:{}".format(class_elbo))
        return class_elbo, logits


def labeled_model():
    x_lab = draw_norm(dim=FLAGS['latent_dim'], mu=x_lab_mu, logvar=x_lab_logvar)
    z2, z2_mu, z2_logvar = q_z2_given_yx(FLAGS, x_lab, y_lab)
    logits = qy_given_x(x_lab, FLAGS)
    x_mu = px_given_z1(FLAGS=FLAGS, y=y_lab, z=z2)
    ELBO = elbo_M2(x_recon=x_mu, x=x_lab, y=y_lab, z=[z2, z2_mu, z2_logvar])
    return ELBO, logits, x_mu


if __name__ == '__main__':
    session = tf.Session()
    # Global Dictionary of Flags
    # learn_yz_x_ss.main(3000, n_labels=100, dataset='mnist_2layer', n_z=50, n_hidden=(300,), seed=seed, alpha=0.1,
    #                    n_minibatches=100, comment='')
    # learn_yz_x_ss.main(3000, n_labels>100, dataset='mnist_2layer', n_z=50, n_hidden=(500,), seed=seed, alpha=0.1,
    #                    n_minibatches=200, comment='')

    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'test_batch_size': 200,
        'num_iterations': 100000,
        'num_batches': 100,
        'seed': 12000,
        'n_labeled': 100,
        'alpha': 0.1,
        'm1_h_dim': 500,
        'm2_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 30000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10
    }
    num_lab_batch, num_ulab_batch, batch_size = get_batch_size(FLAGS)
    np.random.seed(FLAGS['seed'])
    tf.set_random_seed(FLAGS['seed'])
    # data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)
    # ### Placeholder variables
    x_lab_mu = tf.placeholder(tf.float32, shape=[None, FLAGS['latent_dim']], name='x_lab_mu')
    x_unlab_mu = tf.placeholder(tf.float32, shape=[None, FLAGS['latent_dim']], name='x_unlab_mu')
    x_lab_logvar = tf.placeholder(tf.float32, shape=[None, FLAGS['latent_dim']], name='x_ulab_logvar')
    x_unlab_logvar = tf.placeholder(tf.float32, shape=[None, FLAGS['latent_dim']], name='x_unlab_logvar')
    y_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['num_classes']], name='y_lab')
    y_true_cls = tf.argmax(y_lab, axis=1)

    # Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x_l, train_l_y, train_u_x, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(FLAGS['n_labeled'])
    with VAE.session

    # Labeled
    labeled_ELBO, y_lab_logits, x_recon_lab_mu = labeled_model()
    if FLAGS['n_labeled'] == FLAGS['n_train']:
        cost = ((total_lab_loss() * FLAGS['num_batches']) + prior_weights()) / (
            -batch_size * FLAGS['num_batches'])
    else:
        unlabeled_ELBO, y_ulab_logits = unlabeled_model()
        cost = ((total_lab_loss() + total_unlab_loss()) * FLAGS['num_batches'] + prior_weights()) / (
            -batch_size * FLAGS['num_batches'])

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                       beta2=FLAGS['beta2']).minimize(cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    current_dir = os.getcwd()
    save_path = current_dir + "/VAE/semi_supervised/" + FLAGS['summaries_dir'] + 'semi_supervised'
    train_writer = tf.summary.FileWriter(save_path, session.graph)

    train_test()

    session.close()
