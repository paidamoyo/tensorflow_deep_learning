import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from VAE.utils.MNIST_pickled_preprocess import extract_data
from VAE.utils.batch_processing import get_batch_size, get_next_batch
from VAE.utils.distributions import cost_M1
from VAE.utils.metrics import plot_images
from VAE.utils.settings import initialize
from VAE.vanilla.models.decoder import px_given_z1
from VAE.vanilla.models.encoder import q_z1_given_x


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    best_validation_loss = 1e20
    last_improvement = 0

    start_time = time.time()
    idx_labeled = 0
    idx_unlabeled = 0

    for i in range(num_iterations):

        # Batch Training
        x_l_batch, _, idx_labeled = get_next_batch(train_x_l, train_l_y, idx_labeled, num_lab_batch)
        x_u_batch, _, idx_unlabeled = get_next_batch(train_u_x, train_u_y, idx_unlabeled, num_ulab_batch)
        feed_dict_train = {x_lab: x_l_batch, x_unlab: x_u_batch}
        summary, batch_loss, _ = session.run([merged, cost, optimizer], feed_dict=feed_dict_train)
        # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
        train_writer.add_summary(summary, i)

        if (i % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy

            validation_loss = calculate_loss(images=valid_x)
            if validation_loss < best_validation_loss:
                # Save  Best Perfoming all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)
                # update best validation accuracy
                best_validation_loss = validation_loss
                last_improvement = i
                improved_str = '*'
            else:
                improved_str = ''

            print("Optimization Iteration: {}, Training Loss: {}, "
                  " Validation Acc:{}, {}".format(i + 1, batch_loss, validation_loss, improved_str))
        if i - last_improvement > FLAGS['require_improvement']:
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
    # Ending time.
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def calculate_loss(images):
    num_images = len(images)
    total_loss = 0.0
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)
        batch_images = images[i:j, :]
        feed_dict = {x_lab: batch_images}
        batch_loss = session.run(labeled_loss, feed_dict=feed_dict)
        total_loss += batch_loss
        i = j
    return total_loss


def cost_labeled():
    z1, z1_mu, z1_logvar = q_z1_given_x(x_lab)
    x_mu = px_given_z1(z1)
    lab_loss = cost_M1(x_recon=x_mu, x_true=x_lab, z_lsgms=z1_logvar, z_mu=z1_mu)
    return lab_loss, x_mu, z1_mu, z1_logvar


def cost_unlabeled():
    z1, z1_mu, z1_logvar = q_z1_given_x(x_unlab, reuse=True)
    x_mu = px_given_z1(z1, reuse=True)
    unlab_loss = cost_M1(x_recon=x_mu, x_true=x_lab, z_lsgms=z1_logvar, z_mu=z1_mu)
    return unlab_loss, z1_mu, z1_logvar


def train_test():
    train_neural_network(FLAGS['num_iterations'])
    saver.restore(sess=session, save_path=save_path)
    test_reconstruction()


def reconstruct(x_test):
    return session.run(x_recon_lab_mu, feed_dict={x_lab: x_test})


def test_reconstruction():
    num_images = 20
    x_test = test_x[0:num_images, ]
    plot_images(x_test, reconstruct(x_test), num_images, current_dir)


if __name__ == '__main__':
    FLAGS = initialize()
    FLAGS['require_improvement'] = 1000
    session = tf.Session()
    current_dir = os.getcwd()

    num_lab_batch, num_ulab_batch, batch_size = get_batch_size(FLAGS)
    np.random.seed(FLAGS['seed'])
    tf.set_random_seed(FLAGS['seed'])

    x_lab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_labeled')
    x_unlab = tf.placeholder(tf.float32, shape=[None, FLAGS['input_dim']], name='x_unlabeled')

    train_x_l, train_l_y, train_u_x, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(FLAGS['n_labeled'])

    labeled_loss, x_recon_lab_mu, z1_mu_lab, z1_logvar_lab = cost_labeled()
    unlabeled_loss, z1_mu_unlab, z1_logvar_unlab = cost_unlabeled()
    cost = labeled_loss + unlabeled_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                       beta2=FLAGS['beta2']).minimize(cost)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    save_path = current_dir + "VAE/vanilla" + FLAGS['summaries_dir'] + '/VAE'
    train_writer = tf.summary.FileWriter(save_path, session.graph)
    train_test()

    session.close()
