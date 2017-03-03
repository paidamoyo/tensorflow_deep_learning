import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from VAE.classifier import softmax_classifier
from VAE.semi_supervised.decoder import px_given_z1
from VAE.semi_supervised.encoder import q_z2_given_yx, q_z1_given_x
from VAE.utils.MNIST_pickled_preprocess import load_numpy_split, create_semisupervised
from VAE.utils.batch_processing import get_batch_size, get_next_batch
from VAE.utils.distributions import compute_ELBO
from VAE.utils.distributions import prior_weights
from VAE.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images
from VAE.utils.tf_helpers import one_label_tensor

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
        x_l_batch, y_l_batch, idx_labeled = get_next_batch(train_x_l, train_l_y, idx_labeled, num_lab_batch)
        x_u_batch, _, idx_unlabeled = get_next_batch(train_u_x, train_u_y, idx_unlabeled, num_ulab_batch)
        feed_dict_train = {x_lab: x_l_batch, y_lab: y_l_batch, x_unlab: x_u_batch}
        summary, batch_loss, _ = session.run([merged, cost, optimizer], feed_dict=feed_dict_train)
        train_writer.add_summary(summary, batch_loss)

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


def reconstruct(x_test, y_test):
    return session.run(x_recon_lab_mu, feed_dict={x_lab: x_test, y_lab: y_test})


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
    weighted_EBO = tf.reduce_sum(tf.multiply(y_ulab, tf.subtract(unlabeled_ELBO, tf.log(y_lab + const))), 1)
    unlabeled_loss = tf.reduce_sum(weighted_EBO)
    print("unlabeled_ELBO:{}, unlabeled_loss:{}".format(unlabeled_ELBO, unlabeled_loss))
    tf.summary.scalar('unlabeled_loss', unlabeled_loss)
    return unlabeled_loss


def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)
        test_images = images[i:j, :]
        labels = labels[i:j, :]
        feed_dict = {x_lab: test_images,
                     y_lab: labels[i:j, :]}
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
    z1, logits = q_z1_given_x(FLAGS, x_unlab, reuse=True)
    for label in range(FLAGS['num_classes']):
        y_ulab = one_label_tensor(label, num_ulab_batch, FLAGS['num_classes'])
        z2, z2_mu, z2_logvar = q_z2_given_yx(FLAGS, z1, y_ulab, reuse=True)
        x_mu = px_given_z1(FLAGS=FLAGS, y=y_ulab, z=z2, reuse=True)
        _ELBO = tf.expand_dims(compute_ELBO(x_recon=x_mu, x=x_unlab, y=y_ulab, z=[z2, z2_mu, z2_logvar]), 1)
        if label == 0:
            unlabeled_ELBO = tf.identity(_ELBO)
        else:
            unlabeled_ELBO = tf.concat((unlabeled_ELBO, _ELBO), axis=1)  # Decoder Model
    return unlabeled_ELBO, logits


def labeled_model():
    z1, logits = q_z1_given_x(FLAGS, x_lab, reuse=True)
    z2, z2_mu, z2_logvar = q_z2_given_yx(FLAGS, z1, y_lab, reuse=True)
    x_mu = px_given_z1(FLAGS=FLAGS, y=y_lab, z=z2, reuse=True)
    ELBO = compute_ELBO(x_recon=x_mu, x=x_lab, y=y_lab, z=[z2, z2_mu, z2_logvar])
    return ELBO, logits, x_mu


def extract_data():
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy_split(binarize_y=True)
    x_l, y_l, x_u, y_u = create_semisupervised(train_x, train_y, FLAGS['n_labeled'])
    t_x_l, t_y_l = x_l.T, y_l.T
    t_x_u, t_y_u = x_u.T, y_u.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    print("x_l:{}, y_l:{}, x_u:{}, y_{}".format(t_x_l.shape, t_y_l.shape, t_x_u.shape, t_y_u.shape))
    return t_x_l, t_y_l, t_x_u, t_x_u, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':
    session = tf.Session()
    # Global Dictionary of Flags
    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'test_batch_size': 256,
        'num_iterations': 40000,
        'num_batches': 100,
        'seed': 12000,
        'n_labeled': 100,
        'alpha': 0.1,
        'm1_h_dim': 600,
        'm2_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 30000,
        'n_train': 50000,
        'learning_rate': 1e-10,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10,
        'svmC': 1
    }
    num_lab_batch, num_ulab_batch, batch_size = get_batch_size(FLAGS)
    np.random.seed(FLAGS['seed'])
    # data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)
    # ### Placeholder variables
    x_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_labeled')
    x_unlab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_unlabeled')
    y_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['num_classes']], name='y_lab')
    y_true_cls = tf.argmax(y_lab, axis=1)

    # Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x_l, train_l_y, train_u_x, train_u_y, valid_x, valid_y, test_x, test_y = extract_data()

    # Labeled
    labeled_ELBO, y_lab_logits, x_recon_lab_mu = labeled_model()
    unlabeled_ELBO, y_ulab_logits = unlabeled_model()
    cost = ((total_lab_loss() + total_unlab_loss()) * FLAGS['num_batches'] + prior_weights()) / (
        -batch_size * FLAGS['num_batches'])
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                       beta2=FLAGS['beta2']).minimize(cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train', session.graph)

    train_test()

    session.close()
