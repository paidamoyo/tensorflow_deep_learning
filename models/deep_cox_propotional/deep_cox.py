import logging
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index

from models.deep_cox_propotional.faraggi_model import log_hazard_ratio
from models.deep_cox_propotional.simulated_data import SimulatedData
from models.utils.batch_processing import get_last_batch_index
from models.utils.distributions import prior_weights


class DeepCox(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 hidden_dim=500

                 ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.num_classes = 2
        self.num_examples = 50000
        self.min_std = 0.1
        self.log_file = 'deep_cox.log'
        self.batch_norm = True
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.input_dim = 10
        self.batch_norm = True
        self.simulated_data = SimulatedData(num_features=self.input_dim)

        ''' Create Graph '''
        self.G = tf.Graph()
        with self.G.as_default():
            train_data = self.simulated_data.generate_data(N=self.num_examples, method='linear')
            self.train_x, self.train_t, self.train_y = train_data['x'], train_data['t'], train_data['e']

            valid_data = self.simulated_data.generate_data(N=10000, method='linear')
            self.valid_x, self.valid_t, self.valid_y = valid_data['x'], train_data['t'], train_data['e']
            test_data = self.simulated_data.generate_data(N=10000, method='linear')
            self.test_x, self.test_t, self.test_y = test_data['x'], test_data['t'], test_data['e']

            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None], name='y')
            self.t = tf.placeholder(tf.float32, shape=[None], name='t')

            self.is_training = tf.placeholder(tf.bool)
            self._objective()
            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.session = tf.Session()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/deep_cox"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)

    def _objective(self):
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))

        self.neg_lik, self.predicted_time, self.risk = self.negative_log_likelihood()
        self.cost = (self.neg_lik * self.num_examples + prior_weights()) / (
            -self.num_examples)
        tf.summary.scalar('cost', self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def negative_log_likelihood(self):
        """Return the negative log-likelihood of the prediction
            of this model under a given target distribution.

        .. math::

            \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))]
                - \lambda P(\theta)

        where:
            D is the set of observed events
            R_i is the set of examples that are still alive at time of death t_j
            F(x,\theta) = log hazard rate
            P(\theta) = regularization equation
            \lamba = regularization coefficient

        Note: We assume that there are no tied event times

        Parameters:
            E (n,): TensorVector that corresponds to a vector that gives the censor
                variable for each example
            deterministic: True or False. Determines if the output of the network
                is calculated determinsitically.

        Returns:
            neg_likelihood: Theano expression that computes negative
                partial Cox likelihood
        """
        risk = log_hazard_ratio(self.x, self.input_dim, self.hidden_dim, self.batch_norm, self.is_training)
        hazard_ratio = tf.exp(risk)

        log_risk = tf.log(tf.reduce_sum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * self.y
        neg_likelihood = -tf.reduce_sum(censored_likelihood)
        # Relevant Functions
        predicted_time = tf.exp(risk)  # e^h(x)
        print("lik:{}, predicted_time:{}, risk:{}".format(neg_likelihood.shape, predicted_time.shape, risk.shape))
        return neg_likelihood, predicted_time, risk

    def predict_concordance_index(self, x, t, y):
        input_size = len(x)
        i = 0
        num_batches = input_size / self.batch_size
        total_negative_likelihood = 0.0
        total_ci_index = 0.0
        while i < input_size:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, input_size)
            batch_x = x[i:j, :]
            batch_t = t[i:j]
            batch_y = y[i:j]
            feed_dict = {self.x: batch_x,
                         self.t: batch_t,
                         self.y: batch_y,
                         self.is_training: False}
            predicted_time, batch_negative_lik = self.session.run([self.predicted_time, self.neg_lik],
                                                                  feed_dict=feed_dict)
            total_negative_likelihood += batch_negative_lik
            ci_index = concordance_index(batch_t, predicted_time.reshape(batch_y.shape), batch_y)
            total_ci_index += ci_index
            i = j
        return total_ci_index / num_batches, total_negative_likelihood / num_batches

    def train_neural_network(self):
        train_print = "Training Cox Deep Surv:"
        print(train_print)
        logging.debug(train_print)
        self.session.run(tf.global_variables_initializer())

        best_ci = 0
        last_improvement = 0

        start_time = time.time()
        idx = 0
        idx_unlabeled = 0

        for i in range(self.num_iterations):

            # Batch Training
            j = get_last_batch_index(self.train_x.shape[0], idx, self.batch_size)
            x_batch, t_batch, y_batch = self.train_x[idx:j, :], self.train_t[idx:j], self.train_y[idx:j]
            idx = j

            feed_dict_train = {self.x: x_batch, self.t: t_batch, self.y: y_batch, self.is_training: True}

            summary, predicted_time, train_loss, train_neg_lik, _ = self.session.run(
                [self.merged, self.predicted_time, self.cost, self.neg_lik, self.optimizer],
                feed_dict=feed_dict_train)

            train_ci = concordance_index(t_batch, predicted_time.reshape(t_batch.shape), y_batch)

            self.train_writer.add_summary(summary, i)

            if (i % 100 == 0) or (i == (self.num_iterations - 1)):

                # Calculate the CI
                valid_ci, valid_neg_lik = self.predict_concordance_index(x=self.valid_x,
                                                                         y=self.valid_y,
                                                                         t=self.valid_t)
                if valid_ci > best_ci:
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # update best validation accuracy
                    best_ci = valid_ci
                    last_improvement = i
                    improved_str = '*'
                else:
                    improved_str = ''

                optimization_print = "Iteration: {}, Training Loss: {} ci:{} neg_lik: {} " \
                                     " Validation ci:{} marg_lik: {} , {}".format(i + 1, int(train_loss), train_ci,
                                                                                  train_neg_lik,
                                                                                  valid_ci, valid_ci,
                                                                                  improved_str)
                print(optimization_print)
                logging.debug(optimization_print)
                # if i - last_improvement > self.require_improvement:
                #     print("No improvement found in a while, stopping optimization.")
                #     # Break out from the for-loop.
                #     break
        # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        time_dif_print = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(time_dif_print)
        logging.debug(time_dif_print)

    def plot_risk_surface(self, data, risk, i=0, j=1,
                          figsize=(6, 4), x_lims=None, y_lims=None, c_lims=None):
        """
        Plots the predicted risk surface of the network with respect to two
        observed covarites i and j.

        Parameters:
            data: (n,d) numpy array of observations of which to predict risk.
            i: index of data to plot as axis 1
            j: index of data to plot as axis 2
            figsize: size of figure for matplotlib
            x_lims: Optional. If provided, override default x_lims (min(x_i), max(x_i))
            y_lims: Optional. If provided, override default y_lims (min(x_j), max(x_j))
            c_lims: Optional. If provided, override default color limits.

        Returns:
            fig: matplotlib figure object.
        """
        fig = plt.figure(figsize=figsize)
        print("data:{}, risk:{}".format(data.shape, risk.shape))
        X = data[:, i]
        Y = data[:, j]

        if not x_lims is None:
            x_lims = [np.round(np.min(X)), np.round(np.max(X))]
        if not y_lims is None:
            y_lims = [np.round(np.min(Y)), np.round(np.max(Y))]
        if not c_lims is None:
            c_lims = [np.round(np.min(risk)), np.round(np.max(risk))]

        ax = plt.scatter(X, Y, c=risk, edgecolors='none', marker='.')
        ax.set_clim(c_lims)
        plt.colorbar()
        plt.xlim(x_lims)
        plt.ylim(y_lims)
        plt.xlabel('$x_{%d}$' % i, fontsize=18)
        plt.ylabel('$x_{%d}$' % j, fontsize=18)
        plt.savefig("predicted_risk")

        return fig

    def train_test(self):
        self.train_neural_network()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        ci_pred, test_neg_lik = self.predict_concordance_index(x=self.test_x, y=self.test_y, t=self.test_t)
        feed_dict = {self.x: self.test_x,
                     self.t: self.test_t,
                     self.y: self.test_y,
                     self.is_training: False}
        risk = self.session.run(self.risk, feed_dict=feed_dict)
        self.plot_risk_surface(data=self.test_x, risk=risk)

        lik_print = "test neg_likelihood:{}, ci_pred:{}".format(test_neg_lik, ci_pred)
        print(lik_print)
        logging.debug(lik_print)
