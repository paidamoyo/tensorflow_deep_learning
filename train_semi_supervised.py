import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from VAE.classifier import softmax_classifier
from VAE.semi_supervised.decoder import generator_network
from VAE.semi_supervised.encoder import recognition_network
from VAE.utils.MNSIT_prepocess import preprocess_train_data
from VAE.utils.distributions import tf_normal_logpdf, draw_norm
from VAE.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images

sys.path.append(os.getcwd())


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    best_validation_accuracy = 0
    last_improvement = 0

    start_time = time.time()
    x_l, y_l, x_u, y_u = preprocess_train_data(data=data, n_labeled=FLAGS['n_labeled'], n_train=FLAGS['n_train'])

    idx_labeled = 0
    idx_unlabeled = 0

    for epoch in range(num_iterations):

        if np.random.rand() < 0.5:
            batch_loss, j = train_batch(idx_labeled, x_l, y_l, labeled_loss, labeled_optimizer)
            idx_labeled = j
            loss_string = "LABELED"

        else:
            batch_loss, j = train_batch(idx_unlabeled, x_u, y_u, unlabeled_loss, unlabeled_optimizer)
            idx_unlabeled = j
            loss_string = "UNLABELED"

        if (epoch % 100 == 0) or (epoch == (num_iterations - 1)):
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
                last_improvement = epoch
                improved_str = '*'
            else:
                improved_str = ''

            print("Optimization Iteration: {}, {} Training Loss: {}, "
                  " Validation Acc:{}, {}".format(epoch + 1, loss_string, batch_loss, acc_validation, improved_str))
        if epoch - last_improvement > FLAGS['require_improvement']:
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
    # Ending time.
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def train_batch(idx, x_images, y_labels, loss, optimizer):
    # Batch Training
    num_images = x_images.shape[0]
    if idx == num_images:
        idx = 0
    j = min(idx + FLAGS['train_batch_size'], num_images)
    x_batch = x_images[idx:j, :]
    y_true_batch = y_labels[idx:j, :]
    feed_dict_train = {x: x_batch, y_true: y_true_batch}
    summary, batch_loss, _ = session.run([merged, loss, optimizer], feed_dict=feed_dict_train)
    train_writer.add_summary(summary, batch_loss)
    return batch_loss, j


def reconstruct(x_test):
    mean, variance = session.run([x_hat, x_logvar], feed_dict={x: x_test})
    print("mean:{}, variance:{}".format(mean.shape, variance.shape))
    x_re = draw_norm(dim=FLAGS['input_dim'], mu=mean, logvar=variance)
    return x_re


def test_reconstruction():
    x_test = data.test.next_batch(100)[0][0:5, ]
    x_reconstruct = reconstruct(x_test)
    print("x_reconstruct 1:{}".format(x_reconstruct[1]))
    plot_images(x_test, x_reconstruct)


def compute_labeled_loss():
    global y_pred_cls
    # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
    beta = FLAGS['alpha'] * (1.0 * FLAGS['n_labeled'])
    classifier_loss, y_pred_cls = softmax_classifier(logits=y_logits, y_true=y_true)
    # classifier_loss, y_pred_cls = svm_classifier(weights=weights, logits=y_logits, svmC=FLAGS['svmC'],
    #                                              y_true=y_true)
    weighted_classification_loss = beta * classifier_loss
    loss = tf.reduce_mean(
        recognition_loss + reconstruction_loss() + weighted_classification_loss) + y_regularization_loss
    tf.summary.scalar('labeled_loss', loss)
    return loss


def compute_unlabeled_loss():
    # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
    pi = tf.nn.softmax(y_logits)
    entropy = tf.einsum('ij,ij->i', pi, tf.log(pi))
    vae_loss = recognition_loss + reconstruction_loss()
    weighted_loss = tf.einsum('ij,ik->i', tf.reshape(vae_loss, [FLAGS['train_batch_size'], 1]), pi)
    print("entropy:{}, pi:{}, weighted_loss:{}".format(entropy, pi, weighted_loss))
    pi_mean = tf.reduce_mean(pi, axis=0)
    weighted_y_reg = tf.reduce_sum(tf.scalar_mul(scalar=y_regularization_loss, x=pi_mean))
    loss = tf.reduce_mean(weighted_loss) + weighted_y_reg
    tf.summary.scalar('unlabeled_loss', loss)
    return loss


def reconstruction_loss():
    return -tf.reduce_sum(tf_normal_logpdf(x, x_hat, x_logvar), axis=1)


def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)
        test_images = images[i:j, :]
        labels = labels[i:j, :]
        feed_dict = {x: test_images,
                     y_true: labels[i:j, :]}
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


if __name__ == '__main__':
    # Global Dictionary of Flags
    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'train_batch_size': 200,
        'test_batch_size': 256,
        'num_iterations': 2,
        'seed': 12000,
        'n_labeled': 3000,
        'alpha': 0.1,
        'encoder_h_dim': 500,
        'decoder_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 1500,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10,
        'svmC': 1
    }

    np.random.seed(FLAGS['seed'])
    data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)
    # ### Placeholder variables
    x = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, FLAGS['num_classes']], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    # Encoder Model
    z_latent_rep, recognition_loss, y_regularization_loss, y_logits, weights = recognition_network(FLAGS, x)
    # Decoder Model
    x_hat, x_logvar = generator_network(FLAGS=FLAGS, y_logits=y_logits, z_latent_rep=z_latent_rep)
    # Loss and Optimization
    labeled_loss = compute_labeled_loss()
    unlabeled_loss = compute_unlabeled_loss()

    labeled_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                               beta2=FLAGS['beta2']).minimize(labeled_loss)
    unlabeled_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                                 beta2=FLAGS['beta2']).minimize(unlabeled_loss)
    session = tf.Session()
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train', session.graph)

    train_test()

    session.close()
