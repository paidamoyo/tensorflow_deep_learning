import logging
import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from models.classifier import softmax_classifier
from models.auxiliary_semi_supervised.decoder import px_given_zy
from models.auxiliary_semi_supervised.encoder import q_a_given_x, q_z_given_yx, qy_given_ax
from models.utils.MNIST_pickled_preprocess import load_numpy_split, create_semisupervised
from models.utils.batch_processing import get_batch_size, get_next_batch
from models.utils.distributions import compute_ELBO
from models.utils.distributions import prior_weights
from models.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images
from models.utils.tf_helpers import one_label_tensor, variable_summaries


class Auxiliary(object):
    def __init__(self,
                 num_batches,
                 learning_rate,
                 beta1,
                 beta2,
                 alpha,
                 require_improvement,
                 seed,
                 n_labeled,
                 num_iterations,
                 input_dim,
                 latent_dim=100,
                 hidden_dim=500,
                 restore_vae=False
                 ):
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.hidden_dim = hidden_dim
        self.num_batches = num_batches
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.alpha = alpha
        self.n_labeled = n_labeled
        self.num_classes = 10
        self.restore_vae = restore_vae
        self.num_examples = 50000
        self.log_file = 'auxiliary.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ''' Create Graph '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.x_lab = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x_labeled')
            self.x_unlab = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x_unlabeled')
            self.y_lab = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_lab')
            self.y_true_cls = tf.argmax(self.y_lab, axis=1)
            self.train_x_l, self.train_l_y, self.train_u_x, self.train_u_y, self.valid_x, self.valid_y, \
            self.test_x, self.test_y = self.extract_data()
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/auxiliary_semi_supervised_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):

        # Labeled
        self.num_lab_batch, self.num_ulab_batch, self.batch_size = get_batch_size(num_examples=self.num_examples,
                                                                                  num_batches=self.num_batches,
                                                                                  num_lab=self.n_labeled)
        logging.debug(
            "num batches:{}, batch_size:{},  num_lab_batch {}, num_ulab_batch:{}, epochs:{}".format(self.num_batches,
                                                                                                    self.batch_size,
                                                                                                    self.num_lab_batch,
                                                                                                    self.num_ulab_batch,
                                                                                                    int(
                                                                                                        self.num_iterations / self.num_batches)))
        self.labeled_ELBO, self.y_lab_logits, self.x_recon_lab_mu, self.classifier_loss, \
        self.y_pred_cls = self.labeled_model()
        if self.n_labeled == self.num_examples:
            self.train_x_l = np.concatenate((self.train_x_l, self.train_u_x), axis=0)
            self.train_l_y = np.concatenate((self.train_l_y, self.train_u_y), axis=0)
            self.cost = ((self.total_lab_loss() * self.num_batches) + prior_weights()) / (
                -self.num_batches * self.num_batches)
        else:
            self.unlabeled_ELBO, self.y_ulab_logits = self.unlabeled_model()
            self.cost = ((self.total_lab_loss() + self.total_unlab_loss()) * self.num_batches + prior_weights()) / (
                -self.batch_size * self.num_batches)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def extract_data(self):
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy_split(binarize_y=True)
        x_l, y_l, x_u, y_u = create_semisupervised(train_x, train_y, self.n_labeled)
        t_x_l, t_y_l = x_l.T, y_l.T
        t_x_u, t_y_u = x_u.T, y_u.T
        x_valid, y_valid = valid_x.T, valid_y.T
        x_test, y_test = test_x.T, test_y.T

        print("x_l:{}, y_l:{}, x_u:{}, y_{}".format(t_x_l.shape, t_y_l.shape, t_x_u.shape, t_y_u.shape))
        logging.debug("x_l:{}, y_l:{}, x_u:{}, y_{}".format(t_x_l.shape, t_y_l.shape, t_x_u.shape, t_y_u.shape))
        return t_x_l, t_y_l, t_x_u, t_y_u, x_valid, y_valid, x_test, y_test

    def train_neural_network(self):
        print("Training Auxiliary VAE:")
        logging.debug("TTraining Auxiliary VAE:")
        self.session.run(tf.global_variables_initializer())

        best_validation_accuracy = 0
        last_improvement = 0

        start_time = time.time()
        idx_labeled = 0
        idx_unlabeled = 0
        idx = 0

        for i in range(self.num_iterations):

            # Batch Training
            x_l_batch, y_l_batch, idx_labeled = get_next_batch(self.train_x_l, self.train_l_y, idx_labeled,
                                                               self.num_lab_batch)
            x_u_batch, _, idx_unlabeled = get_next_batch(self.train_u_x, self.train_u_y, idx_unlabeled,
                                                         self.num_ulab_batch)
            feed_dict_train = {self.x_lab: x_l_batch, self.y_lab: y_l_batch,
                               self.x_unlab: x_u_batch}

            summary, batch_loss, _ = self.session.run([self.merged, self.cost, self.optimizer],
                                                      feed_dict=feed_dict_train)
            # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
            self.train_writer.add_summary(summary, i)

            if (i % 100 == 0) or (i == (self.num_iterations - 1)):
                # Calculate the accuracy
                correct, _, log_lik = self.predict_cls(images=self.valid_x,
                                                       labels=self.valid_y,
                                                       cls_true=convert_labels_to_cls(self.valid_y))
                acc_validation, _ = cls_accuracy(correct)
                if acc_validation > best_validation_accuracy:
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # update best validation accuracy
                    best_validation_accuracy = acc_validation
                    last_improvement = i
                    improved_str = '*'
                else:
                    improved_str = ''

                print("Iteration: {}, Training Loss: {}, "
                      " Validation:  log_lik {},  Acc {}, {}".format(i + 1, int(batch_loss), int(log_lik),
                                                                     acc_validation,
                                                                     improved_str))
                logging.debug("Iteration: {}, Training Loss: {}, "
                              " Validation:  log_lik {},  Acc {}, {}".format(i + 1, int(batch_loss), int(log_lik),
                                                                             acc_validation,
                                                                             improved_str))
            if i - last_improvement > self.require_improvement:
                print("No improvement found in a while, stopping optimization.")
                # Break out from the for-loop.
                break
        # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        logging.debug("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def reconstruct(self, x_test, y_test):
        return self.session.run(self.x_recon_lab_mu, feed_dict={self.x: x_test, self.x_lab: x_test, self.y_lab: y_test})

    def test_reconstruction(self):
        num_images = 20
        x_test = self.test_x[0:num_images, ]
        y_test = self.test_y[0:num_images, ]
        plot_images(x_test, self.reconstruct(x_test, y_test), num_images, "pre_semi_supervised")

    def total_lab_loss(self):
        # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
        beta = self.alpha * (float(self.batch_size) / self.num_lab_batch)
        weighted_classifier_loss = beta * self.classifier_loss
        labeled_loss = tf.reduce_sum(tf.subtract(self.labeled_ELBO, weighted_classifier_loss))
        tf.summary.scalar('labeled_loss', labeled_loss)
        return labeled_loss

    def total_unlab_loss(self):
        # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
        const = 1e-10
        y_ulab = tf.nn.softmax(logits=self.y_ulab_logits)
        variable_summaries(self.y_lab, 'y_lab')
        weighted_elbo = tf.reduce_sum(
            tf.multiply(y_ulab + const, tf.subtract(self.unlabeled_ELBO, tf.log(y_ulab + const))),
            1)
        unlabeled_loss = tf.reduce_sum(weighted_elbo)
        print("unlabeled_ELBO:{}, unlabeled_loss:{}".format(self.unlabeled_ELBO, unlabeled_loss))
        tf.summary.scalar('unlabeled_loss', unlabeled_loss)
        return unlabeled_loss

    def predict_cls(self, images, labels, cls_true):
        num_images = len(images)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        total_log_lik = 0.0
        i = 0
        num_val_batches = int(10000 / self.batch_size)
        mean_value, update_op = tf.contrib.metrics.streaming_auc(self.y_lab_logits, self.y_lab, curve='ROC')
        final_mean_value = 0.0
        self.session.run(tf.local_variables_initializer())
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, num_images)
            batch_images = images[i:j, :]
            batch_labels = labels[i:j, :]
            feed_dict = {self.x_lab: batch_images,
                         self.x: batch_images,
                         self.y_lab: batch_labels}
            cls_pred[i:j], log_lik = self.session.run([self.y_pred_cls, self.log_lik],
                                                      feed_dict=feed_dict)
            total_log_lik += log_lik
            i = j
            final_mean_value = mean_value.eval(feed_dict=feed_dict)
        print('Final Mean AUC: %f' % final_mean_value)
        logging.debug('Final Mean AUC: %f' % final_mean_value)
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)
        return correct, cls_pred, total_log_lik / num_val_batches

    def train_test(self):
        self.train_neural_network()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        correct, cls_pred, _ = self.predict_cls(images=self.test_x,
                                                labels=self.test_y,
                                                cls_true=(convert_labels_to_cls(self.test_y)))
        print_test_accuracy(correct, cls_pred, self.test_y, logging)
        self.test_reconstruction()

    def unlabeled_model(self):
        # Ulabeled
        a, a_mu, a_logvar = q_a_given_x(self.x_unlab, hidden_dim=self.hidden_dim, input_dim=self.input_dim,
                                            latent_dim=self.latent_dim, reuse=True)
        logits = qy_given_ax(a=a, x=self.x_unlab, latent_dim=self.latent_dim,
                             num_classes=self.num_classes, hidden_dim=self.hidden_dim, input_dim=self.input_dim)
        for label in range(self.num_classes):
            y_ulab = one_label_tensor(label, self.num_ulab_batch, self.num_classes)
            z2, z2_mu, z2_logvar = q_z2_given_z1y(z1=z1, y=y_ulab, latent_dim=self.latent_dim,
                                                  num_classes=self.num_classes, hidden_dim=self.hidden_dim, reuse=True)
            z1_recon, z1_mu_recon, z1_var_recon = pz1_given_z2y(y=y_ulab, z2=z2, latent_dim=self.latent_dim,
                                                                num_classes=self.num_classes,
                                                                hidden_dim=self.hidden_dim,
                                                                reuse=True)
            x_recon_mu = px_given_z1(z1_recon, latent_dim=self.latent_dim,
                                     hidden_dim=self.hidden_dim, input_dim=self.input_dim, reuse=True)
            _elbo = tf.expand_dims(compute_ELBO(x_recon=x_recon_mu, x=self.x_unlab, y=y_ulab, z=[z2, z2_mu, z2_logvar]),
                                   1)

            if label == 0:
                class_elbo = tf.identity(_elbo)
            else:
                class_elbo = tf.concat((class_elbo, _elbo), axis=1)  # Decoder Model
        print("unlabeled class_elbo:{}".format(class_elbo))
        return class_elbo, logits

    def labeled_model(self):
        a, a_mu, a_logvar = q_a_given_x(self.x_lab, hidden_dim=self.hidden_dim, input_dim=self.input_dim,
                                            latent_dim=self.latent_dim, reuse=True)


        logits = qy_given_ax(a=a,x=self.x_lab, latent_dim=self.latent_dim,
                             num_classes=self.num_classes, hidden_dim=self.hidden_dim, input_dim=self.input_dim)

        z2, z2_mu, z2_logvar = q_z_given_yx(z1=z1, y=self.y_lab, latent_dim=self.latent_dim,
                                              num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        z1_recon, z1_mu_recon, z1_var_recon = pz1_given_z2y(y=self.y_lab, z2=z2, latent_dim=self.latent_dim,
                                                            num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        x_recon_mu = px_given_z1(z1_recon, latent_dim=self.latent_dim,
                                 hidden_dim=self.hidden_dim, input_dim=self.input_dim, reuse=True)
        elbo = compute_ELBO(x_recon=x_recon_mu, x=self.x_lab, y=self.y_lab, z=[z2, z2_mu, z2_logvar])

        classifier_loss, y_pred_cls = softmax_classifier(logits=logits, y_true=self.y_lab)
        return elbo, logits, x_recon_mu, classifier_loss, y_pred_cls
