import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from models.utils.MNIST_pickled_preprocess import extract_data
from models.utils.batch_processing import get_next_batch
from models.utils.distributions import elbo_M1, prior_weights
from models.utils.metrics import plot_images
from models.vanilla_vae.decoder import px_given_z1
from models.vanilla_vae.encoder import q_z1_given_x


class VariationalAutoencoder(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 input_dim, latent_dim,
                 hidden_dim=600,
                 ):
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ''' Create Graph '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/vae_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):
        n_train_examples = 50000
        train_x_l, train_l_y, train_u_x, train_u_y, self.valid_x, self.valid_y, self.test_x, self.test_y = extract_data(
            n_train_examples)
        num_batches = int(n_train_examples / self.batch_size)
        self.train_x = np.concatenate((train_x_l, train_u_x), axis=0)
        self.train_y = np.concatenate((train_l_y, train_u_y), axis=0)
        elbo, self.x_recon_mu, self.z_sample, self.z_mu, self.z_logvar = self.build_model()
        self.cost = (elbo * num_batches + prior_weights()) / (-self.batch_size * num_batches)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def train(self):
        print("Training Vanilla VAE:")
        self.session.run(tf.global_variables_initializer())
        best_validation_loss = 1e20
        last_improvement = 0

        start_time = time.time()
        idx = 0

        for i in range(self.num_iterations):
            # Batch Training
            x_batch, _, idx = get_next_batch(self.train_x, self.train_y, idx, self.batch_size)
            summary, batch_loss, _ = self.session.run([self.merged, self.cost, self.optimizer],
                                                      feed_dict={self.x: x_batch})
            # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
            self.train_writer.add_summary(summary, i)

            if (i % 100 == 0) or (i == (self.num_iterations - 1)):
                # Calculate the accuracy

                validation_loss = self.calculate_loss(images=self.valid_x)
                if validation_loss < best_validation_loss:
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # update best validation accuracy
                    best_validation_loss = validation_loss
                    last_improvement = i
                    improved_str = '*'

                else:
                    improved_str = ''

                print("Optimization Iteration: {}, Training Loss: {}, "
                      " Validation Loss:{}, {}".format(i + 1, batch_loss, validation_loss, improved_str))
            if i - last_improvement > self.require_improvement:
                print("No improvement found in a while, stopping optimization.")
                # Break o    ut from the for-loop.
                break  # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def calculate_loss(self, images):
        num_images = len(images)
        total_loss = 0.0
        i = 0
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, num_images)
            batch_images = images[i:j, :]
            feed_dict = {self.x: batch_images}
            batch_loss = self.session.run(self.cost, feed_dict=feed_dict)
            total_loss += batch_loss
            i = j
        return total_loss

    def build_model(self):
        z, z_mu, z_logvar = q_z1_given_x(self.x, hidden_dim=self.hidden_dim, input_dim=self.input_dim,
                                         latent_dim=self.latent_dim)
        x_mu = px_given_z1(z1=z, hidden_dim=self.hidden_dim, input_dim=self.input_dim,
                           latent_dim=self.latent_dim)
        loss = elbo_M1(x_recon=x_mu, x_true=self.x, z=z, z_lsgms=z_logvar, z_mu=z_mu)
        return tf.reduce_sum(loss), x_mu, z, z_mu, z_logvar

    def train_test(self):
        self.train()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        self.test_reconstruction()

    def test_reconstruction(self):
        num_images = 20
        x_test = self.test_x[0:num_images, ]
        plot_images(x_test, self.decode(x_test), num_images, "vae")

    def decode(self, x_test):
        return self.session.run(self.x_recon_mu, feed_dict={self.x: x_test})

    def encode(self, x_input, sample=False):
        if sample:
            return self.session.run([self.z_sample, self.z_mu, self.z_logvar], feed_dict={self.x: x_input})
        else:
            return self.session.run([self.z_mu, self.z_logvar], feed_dict={self.x: x_input})
