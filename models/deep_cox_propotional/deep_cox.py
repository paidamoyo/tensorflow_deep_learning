import logging
import os

import numpy as np
import tensorflow as tf

from models.deep_cox_propotional.data.simulated import SimulatedData
from models.utils.distributions import prior_weights


class DeepCox(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 alpha,
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
        self.alpha = alpha
        self.num_classes = 10
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
            self.train_x, self.train_t, self.train_y = train_data['x'], train_data['y'], train_data['e']
            valid_data = self.simulated_data.generate_data(N=10000, method='linear')
            self.valid_x, self.valid_t, self.valid_y = train_data['x'], train_data['y'], train_data['e']
            self.test = self.simulated_data.generate_data(N=10000, method='linear')
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
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

        self.cost = (self.total_likelihood() * self.num_examples + prior_weights()) / (-self.num_examples)
        tf.summary.scalar('cost', self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def _negative_log_likelihood(self, E, deterministic=False):
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
        risk = self.risk(deterministic)
        hazard_ratio = T.exp(risk)
        log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
        uncensored_likelihood = risk.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -T.sum(censored_likelihood)
        return neg_likelihood
