import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from models.classifier import softmax_classifier
from models.semi_supervised_vae.decoder import px_given_zy
from models.semi_supervised_vae.encoder import q_z_given_xy, qy_given_x
from models.utils.batch_processing import get_encoded_next_batch, get_batch_size
from models.utils.distributions import draw_norm
from models.utils.distributions import elbo_M2
from models.utils.distributions import prior_weights
from models.utils.metrics import cls_accuracy, print_test_accuracy, convert_labels_to_cls, plot_images
from models.utils.tf_helpers import one_label_tensor, variable_summaries


class GenerativeClassifier(object):
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
                 input_dim, latent_dim,
                 train_lab,
                 train_unlab,
                 valid,
                 test,
                 hidden_dim=600
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
        self.train_x_l_mu, self.train_x_l_logvar, self.train_l_y = train_lab[0], train_lab[1], train_lab[2]
        self.train_x_u_mu, self.train_x_u_logvar, self.train_u_y = train_unlab[0], train_unlab[1], train_unlab[2]
        self.valid_x_mu, self.valid_x_logvar, self.valid_y = valid[0], valid[1], valid[2]
        self.test_x_mu, self.test_x_logvar, self.test_y = test[0], test[1], test[2]
        self.num_classes = 10
        self.num_examples = 50000
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ''' Create Graph '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.x_lab_mu = tf.placeholder(tf.float32, shape=[None, latent_dim], name='x_lab_mu')
            self.x_unlab_mu = tf.placeholder(tf.float32, shape=[None, latent_dim], name='x_unlab_mu')
            self.x_lab_logvar = tf.placeholder(tf.float32, shape=[None, latent_dim], name='x_ulab_logvar')
            self.x_unlab_logvar = tf.placeholder(tf.float32, shape=[None, latent_dim], name='x_unlab_logvar')
            self.y_lab = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_lab')
            self.y_true_cls = tf.argmax(self.y_lab, axis=1)
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/semi_supervised_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):

        # Labeled
        self.num_lab_batch, self.num_ulab_batch, self.batch_size = get_batch_size(num_examples=self.num_examples,
                                                                                  num_batches=self.num_batches,
                                                                                  num_lab=self.n_labeled)
        self.labeled_ELBO, self.y_lab_logits, self.x_recon_lab_mu, self.classifier_loss, self.y_pred_cls = self.labeled_model()
        if self.n_labeled == self.num_examples:
            self.cost = ((self.total_lab_loss() * self.num_batches) + prior_weights()) / (
                -self.num_batches * self.num_batches)
        else:
            self.unlabeled_ELBO, self.y_ulab_logits = self.unlabeled_model()
            self.cost = ((self.total_lab_loss() + self.total_unlab_loss()) * self.num_batches + prior_weights()) / (
                -self.batch_size * self.num_batches)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def train_neural_network(self):
        print("Training Semisupervised VAE:")
        self.session.run(tf.global_variables_initializer())
        best_validation_accuracy = 0
        last_improvement = 0

        start_time = time.time()
        idx_labeled = 0
        idx_unlabeled = 0

        for i in range(self.num_iterations):

            # Batch Training
            x_l_mu, x_l_logvar, y_l_batch, idx_labeled = get_encoded_next_batch(self.train_x_l_mu,
                                                                                self.train_x_l_logvar,
                                                                                self.train_l_y,
                                                                                idx_labeled,
                                                                                self.num_lab_batch)
            x_u_mu, x_u_logvar, _, idx_unlabeled = get_encoded_next_batch(self.train_x_u_mu, self.train_x_u_logvar,
                                                                          self.train_u_y,
                                                                          idx_unlabeled,
                                                                          self.num_ulab_batch)
            feed_dict_train = {self.x_lab_mu: x_l_mu, self.y_lab: y_l_batch, self.x_unlab_mu: x_u_mu,
                               self.x_lab_logvar: x_l_logvar,
                               self.x_unlab_logvar: x_u_logvar}
            summary, batch_loss, _ = self.session.run([self.merged, self.cost, self.optimizer],
                                                      feed_dict=feed_dict_train)
            # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
            self.train_writer.add_summary(summary, i)

            if (i % 100 == 0) or (i == (self.num_iterations - 1)):
                # Calculate the accuracy
                correct, _ = self.predict_cls(mu=self.valid_x_mu,
                                              logvar=self.valid_x_logvar,
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

                print("Optimization Iteration: {}, Training Loss: {}, "
                      " Validation Acc:{}, {}".format(i + 1, batch_loss, acc_validation, improved_str))
            if i - last_improvement > self.require_improvement:
                print("No improvement found in a while, stopping optimization.")
                # Break out from the for-loop.
                break
        # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def reconstruct(self, test_mu, test_logvar, y_test):
        return self.session.run(self.x_recon_lab_mu,
                                feed_dict={self.x_lab_mu: test_mu, self.x_lab_logvar: test_logvar, self.y_lab: y_test})

    def test_reconstruction(self):
        num_images = 20
        mu = self.test_x_mu[0:num_images, ]
        logvar = self.test_x_logvar[0:num_images, ]
        y_test = self.test_y[0:num_images, ]
        plot_images(mu, self.reconstruct(test_mu=mu, test_logvar=logvar, y_test=y_test), n_images=num_images,
                    name="semi_supervised")

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
            tf.multiply(y_ulab + const, tf.subtract(self.unlabeled_ELBO, tf.log(self.y_lab + const))),
            1)
        unlabeled_loss = tf.reduce_sum(weighted_elbo)
        print("unlabeled_ELBO:{}, unlabeled_loss:{}".format(self.unlabeled_ELBO, unlabeled_loss))
        tf.summary.scalar('unlabeled_loss', unlabeled_loss)
        return unlabeled_loss

    def predict_cls(self, mu, logvar, labels, cls_true):
        num_images = len(mu)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        i = 0
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, num_images)
            batch_mu = mu[i:j, :]
            batch_logavar = logvar[i:j, :]
            batch_labels = labels[i:j, :]
            feed_dict = {self.x_lab_mu: batch_mu,
                         self.x_lab_logvar: batch_logavar,
                         self.y_lab: batch_labels}
            cls_pred[i:j] = self.session.run(self.y_pred_cls, feed_dict=feed_dict)
            i = j
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)
        return correct, cls_pred

    def train_test(self):
        self.train_neural_network()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        correct, cls_pred = self.predict_cls(mu=self.test_x_mu,
                                             logvar=self.test_x_logvar,
                                             labels=self.test_y,
                                             cls_true=(convert_labels_to_cls(self.test_y)))
        print_test_accuracy(correct, cls_pred, self.test_y)

    def unlabeled_model(self):
        # Ulabeled
        x_unlab = draw_norm(dim=self.latent_dim, mu=self.x_unlab_mu, logvar=self.x_unlab_logvar)
        logits = qy_given_x(x_unlab, latent_dim=self.latent_dim,
                            num_classes=self.num_classes, hidden_dim=self.hidden_dim, reuse=True)
        for label in range(self.num_classes):
            y_ulab = one_label_tensor(label, self.num_ulab_batch, self.num_classes)
            z, z_mu, z_logvar = q_z_given_xy(x=x_unlab, y=y_ulab, latent_dim=self.latent_dim,
                                             num_classes=self.num_classes, hidden_dim=self.hidden_dim, reuse=True)
            x_mu = px_given_zy(y=y_ulab, z=z, latent_dim=self.latent_dim,
                                         num_classes=self.num_classes, hidden_dim=self.hidden_dim,
                                         reuse=True)
            _elbo = tf.expand_dims(elbo_M2(x_recon=[x_mu], x=x_unlab, y=y_ulab, z=[z, z_mu, z_logvar]), 1)

            if label == 0:
                class_elbo = tf.identity(_elbo)
            else:
                class_elbo = tf.concat((class_elbo, _elbo), axis=1)  # Decoder Model
        print("unlabeled class_elbo:{}".format(class_elbo))
        return class_elbo, logits

    def labeled_model(self):
        x_lab = draw_norm(dim=self.latent_dim, mu=self.x_lab_mu, logvar=self.x_lab_logvar)
        z, z_mu, z_logvar = q_z_given_xy(x=x_lab, y=self.y_lab, latent_dim=self.latent_dim,
                                         num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        logits = qy_given_x(x=x_lab, latent_dim=self.latent_dim,
                            num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        x_mu = px_given_zy(y=self.y_lab, z=z, latent_dim=self.latent_dim,
                                     num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        elbo = elbo_M2(x_recon=[x_mu], x=x_lab, y=self.y_lab, z=[z, z_mu, z_logvar])
        classifier_loss, y_pred_cls = softmax_classifier(logits=logits, y_true=self.y_lab)
        return elbo, logits, x_mu, classifier_loss, y_pred_cls
