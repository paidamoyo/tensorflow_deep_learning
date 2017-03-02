import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from VAE.classifier import softmax_classifier
from VAE.semi_supervised.decoder import px_given_z1
from VAE.semi_supervised.encoder import q_z2_given_yx, q_z1_given_x
from VAE.utils.MNSIT_prepocess import preprocess_train_data, get_batch_size
from VAE.utils.distributions import compute_ELBO
from VAE.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images
from VAE.utils.tf_helpers import one_label_tensor

sys.path.append(os.getcwd())


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    best_validation_accuracy = 0
    last_improvement = 0

    start_time = time.time()
    x_l, y_l, x_u, y_u = preprocess_train_data(data=data, n_labeled=FLAGS['n_labeled'], n_train=FLAGS['n_train'])
    idx_labeled = 0
    idx_unlabeled = 0

    for i in range(num_iterations):

        # Batch Training
        x_l_batch, y_l_batch, idx_labeled = get_next_batch(x_l, y_l, idx_labeled, num_lab_batch)
        x_u_batch, _, idx_unlabeled = get_next_batch(x_u, y_u, idx_unlabeled, num_ulab_batch)
        feed_dict_train = {x_lab: x_l_batch, y_lab: y_l_batch, x_unlab: x_u_batch}
        summary, batch_loss, _ = session.run([merged, cost, optimizer], feed_dict=feed_dict_train)
        train_writer.add_summary(summary, batch_loss)

        if (i % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy
            correct, _ = predict_cls(images=data.validation.images,
                                     labels=data.validation.labels,
                                     cls_true=convert_labels_to_cls(data.validation.labels))
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


def get_next_batch(x_images, y_labels, idx, batch_size):
    num_images = x_images.shape[0]
    if idx == num_images:
        idx = 0
    j = min(idx + batch_size, num_images)
    x_batch = x_images[idx:j, :]
    y_true_batch = y_labels[idx:j, :]
    return x_batch, y_true_batch, j


def reconstruct(x_test, y_test):
    return session.run(x_recon_lab_mu, feed_dict={x_lab: x_test, y_lab: y_test})


def test_reconstruction():
    num_images = 20
    batch = data.test.next_batch(100)
    x_test = batch[0][0:num_images, ]
    y_test = batch[1][0:num_images, ]
    plot_images(x_test, reconstruct(x_test, y_test), num_images)


def total_lab_loss():
    global y_pred_cls
    # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
    beta = FLAGS['alpha'] * (float(batch_size) / num_lab_batch)
    classifier_loss, y_pred_cls = softmax_classifier(logits=y_lab_logits, y_true=y_lab)
    weighted_classifier_loss = beta * classifier_loss
    labeled_KL = tf.scalar_mul(scalar=-1, x=labeled_ELBO)
    labeled_loss = tf.reduce_sum(tf.add(labeled_KL, weighted_classifier_loss))
    tf.summary.scalar('labeled_loss', labeled_loss)
    return labeled_loss


def total_unlab_loss():
    # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
    y_ulab = tf.nn.softmax(logits=y_ulab_logits)
    weighted_ELBO = tf.reduce_sum(tf.multiply(y_ulab, tf.subtract(unlabeled_ELBO, tf.log(y_ulab))), 1)
    print("unlabeled_ELBO:{}, weighted_ELBO:{}".format(unlabeled_ELBO, weighted_ELBO))
    unlabeled_KL = tf.scalar_mul(scalar=-1, x=weighted_ELBO)
    unlabeled_loss = tf.reduce_sum(unlabeled_KL)
    tf.summary.scalar('unlabeled_loss', unlabeled_loss)
    return unlabeled_loss


def reconstruction_loss(x_input, x_hat):
    return tf.reduce_sum(tf.squared_difference(x_input, x_hat), axis=1)


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
    correct, cls_pred = predict_cls(images=data.test.images,
                                    labels=data.test.labels,
                                    cls_true=(convert_labels_to_cls(data.test.labels)))
    print_test_accuracy(correct, cls_pred, data.test.labels)
    test_reconstruction()


def unlabeled_model():
    # Ulabeled
    z1_ulab, y_ulab_logits = q_z1_given_x(FLAGS, x_unlab, reuse=True)
    for label in range(FLAGS['num_classes']):
        _y_ulab = one_label_tensor(label, num_ulab_batch, FLAGS['num_classes'])
        print('_y_ulabel:{}, label:{}'.format(_y_ulab, label))
        z2_ulab, z2_ulab_mu, z2_ulab_logvar = q_z2_given_yx(FLAGS, z1_ulab, _y_ulab, reuse=True)
        x_recon_ulab_mu, x_recon_ulab_logvar = px_given_z1(FLAGS=FLAGS, y=_y_ulab,
                                                           z=z2_ulab, reuse=True)
        print("x_recon_ulab_mu:{}, x_recon_ulab_logvar:{}, z2_ulab:{}".format(x_recon_ulab_mu, x_recon_ulab_logvar,
                                                                              z2_ulab))
        _ELBO = tf.expand_dims(
            compute_ELBO(x_recon=[x_recon_ulab_mu, x_recon_ulab_logvar], x=x_unlab, y=_y_ulab,
                         z=[z2_ulab, z2_ulab_mu, z2_ulab_logvar])
            , 1)
        if label == 0:
            unlabeled_ELBO = tf.identity(_ELBO)
        else:
            unlabeled_ELBO = tf.concat((unlabeled_ELBO, _ELBO), axis=1)  # Decoder Model
    return unlabeled_ELBO, y_ulab_logits


def labeled_model():
    z1_lab, y_lab_logits = q_z1_given_x(FLAGS, x_lab, reuse=True)
    z2_lab, z2_lab_mu, z2_lab_logvar = q_z2_given_yx(FLAGS, z1_lab, y_lab, reuse=True)
    x_recon_lab_mu, x_recon_lab_logvar = px_given_z1(FLAGS=FLAGS, y=y_lab, z=z2_lab,
                                                     reuse=True)
    labeled_ELBO = compute_ELBO(x_recon=[x_recon_lab_mu, x_recon_lab_logvar], x=x_lab,
                                y=y_lab,
                                z=[z2_lab, z2_lab_mu, z2_lab_logvar])
    return labeled_ELBO, y_lab_logits, x_recon_lab_mu


if __name__ == '__main__':
    session = tf.Session()
    # Global Dictionary of Flags
    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'test_batch_size': 256,
        'num_iterations': 20000,
        'num_batches': 100,
        'seed': 12000,
        'n_labeled': 100,
        'alpha': 0.1,
        'm1_h_dim': 600,
        'm2_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 2000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10,
        'svmC': 1
    }
    num_lab_batch, num_ulab_batch, batch_size = get_batch_size()
    np.random.seed(FLAGS['seed'])
    data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)
    # ### Placeholder variables
    x_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_labeled')
    x_unlab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_unlabeled')
    y_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['num_classes']], name='y_lab')
    y_true_cls = tf.argmax(y_lab, axis=1)

    # Labeled
    labeled_ELBO, y_lab_logits, x_recon_lab_mu = labeled_model()
    unlabeled_ELBO, y_ulab_logits = unlabeled_model()
    cost = ((total_lab_loss() + total_unlab_loss()) * FLAGS['num_batches']) / (
        batch_size * FLAGS['num_batches'])
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                       beta2=FLAGS['beta2']).minimize(cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train', session.graph)

    train_test()

    session.close()
