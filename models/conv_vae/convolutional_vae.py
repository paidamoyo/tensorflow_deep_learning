import logging
import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from models.conv_vae.decoder import px_given_z1
from models.conv_vae.encoder import q_z1_given_x
from models.utils.MNIST_pickled_preprocess import extract_data
from models.utils.batch_processing import get_next_batch
from models.utils.distributions import elbo_M1, prior_weights
from models.utils.metrics import plot_images, plot_cost


class ConvVariationalAutoencoder(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 input_dim,
                 latent_dim,
                 filter_sizes,
                 fc_size,
                 num_filters,
                 batch_norm=False,
                 keep_prob=1,
                 gpu_memory_fraction=1
                 ):
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.filter_sizes = filter_sizes
        self.fc_size = fc_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.keep_prob = keep_prob
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.log_file = 'conv_vae.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

        self.current_dir = os.getcwd()
        self.save_path = self.current_dir + "/summaries/vae_model"
        self.validation_cost = []
        self.validation_log_lik = []
        self.train_cost = []
        self.train_log_lik = []
        self._build_graph()

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session(config=self.config)
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):
        self.num_examples = 50000
        train_x_l, train_l_y, train_u_x, train_u_y, self.valid_x, self.valid_y, self.test_x, self.test_y = extract_data(
            self.num_examples)
        self.num_batches = int(self.num_examples / self.batch_size)
        logging.debug("num batches:{}, batch_size:{}, epochs:{}".format(self.num_batches, self.batch_size, int(
            self.num_iterations / self.num_batches)))
        self.train_x = np.concatenate((train_x_l, train_u_x), axis=0)
        self.train_y = np.concatenate((train_l_y, train_u_y), axis=0)
        elbo, self.x_recon_mu, self.z_sample, self.z_mu, self.z_logvar, self.loglik = self.build_model()
        self.cost = (elbo * self.num_batches + prior_weights()) / (-self.batch_size * self.num_batches)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def train(self):
        train_print = "Training Conv VAE Model:"
        params_print = "Parameters: filter_sizes:{}, num_filters:{}, learning_rate:{}," \
                       " momentum: beta1={} beta2={}, batch_size:{}, batch_norm:{}," \
                       " latent_dim:{} num_of_batches:{}, keep_prob:{}, fc_size:{}, require_improvement:{}" \
            .format(self.filter_sizes, self.num_filters, self.learning_rate, self.beta1, self.beta2,
                    self.batch_size, self.batch_norm, self.latent_dim, self.num_batches, self.keep_prob,
                    self.fc_size, self.require_improvement)
        print(train_print)
        print(params_print)
        logging.debug(train_print)
        logging.debug(params_print)
        self.session.run(tf.global_variables_initializer())
        best_validation_loss = 1e20
        last_improvement = 0

        start_time = time.time()
        idx = 0
        epochs = 0
        for i in range(self.num_iterations):
            # Batch Training
            x_batch, _, idx = get_next_batch(self.train_x, self.train_y, idx, self.batch_size)
            summary, batch_loss, batch_log_lik, _ = self.session.run(
                [self.merged, self.cost, self.loglik, self.optimizer],
                feed_dict={self.x: x_batch})
            # Batch Trainin
            if idx == self.num_examples:
                epochs += 1
                is_epoch = True
                idx = 0
            else:
                is_epoch = False
            # print("Optimization Iteration: {}, Training Loss: {}".format(i, batch_loss))
            self.train_writer.add_summary(summary, i)

            validation_loss, val_log_lik = self.validation_loss(images=self.valid_x)
            if (is_epoch) or (i == (self.num_iterations - 1)):
                self.train_log_lik.append(batch_log_lik)
                self.train_cost.append(batch_loss)
                self.validation_cost.append(validation_loss)
                self.validation_log_lik.append(val_log_lik)
                # Calculate the accuracy
                if validation_loss < best_validation_loss:
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # update best validation accuracy
                    best_validation_loss = validation_loss
                    last_improvement = i
                    improved_str = '*'

                else:
                    improved_str = ''

                print("Epochs: {}, Training:  Loss {}, batch_log_lik {}"
                      " Validation: Loss {}, batch_log_lik {} {}".format(epochs, int(batch_loss), int(batch_log_lik),
                                                                         int(validation_loss),
                                                                         int(val_log_lik), improved_str))
                logging.debug("Iteration: {}, Training:  Loss {}, batch_log_lik {}"
                              " Validation: Loss {}, batch_log_lik {} {}".format(i + 1, int(batch_loss),
                                                                                 int(batch_log_lik),
                                                                                 int(validation_loss),
                                                                                 int(val_log_lik), improved_str))
            if i - last_improvement > self.require_improvement:
                print("No improvement found in a while, stopping optimization.")
                logging.debug("No improvement found in a while, stopping optimization.")
                # Break o    ut from the for-loop.
                break  # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        logging.debug("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return epochs, last_improvement

    def validation_loss(self, images):
        num_images = len(images)
        total_loss = 0.0
        total_log_lik = 0.0
        i = 0
        num_val_batches = int(num_images / self.batch_size)
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, num_images)
            batch_images = images[i:j, :]
            feed_dict = {self.x: batch_images}
            batch_loss, log_lik = self.session.run([self.cost, self.loglik], feed_dict=feed_dict)
            total_loss += batch_loss
            total_log_lik += log_lik
            i = j
        return total_loss / num_val_batches, total_log_lik / num_val_batches

    def build_model(self):
        print("x_image:{}".format(self.x_image.shape))
        z, z_mu, z_logvar = q_z1_given_x(self.x_image, num_channels=1, filter_sizes=self.filter_sizes,
                                         num_filters=self.num_filters,
                                         fc_size=self.fc_size,
                                         latent_dim=self.latent_dim)
        print("filter_sizes:{},num_filters:{} ".format(self.filter_sizes, self.num_filters))
        # dencov
        self.filter_sizes.reverse()
        self.num_filters.reverse()
        print("reversed: filter_sizes:{},num_filters:{} ".format(self.filter_sizes, self.num_filters))
        print("z:{}".format(z.shape))
        z = tf.expand_dims(tf.expand_dims(z, 1), 1)
        print("expanded z:{}".format(z.shape))
        x_mu = px_given_z1(z1=z, input_dim=self.input_dim,
                           num_channels=50,
                           filter_sizes=self.filter_sizes,
                           fc_size=self.fc_size, num_filters=self.num_filters)
        loss, log_lik = elbo_M1(x_recon=x_mu, x_true=self.x, z1=z, z1_lsgms=z_logvar, z1_mu=z_mu)
        return tf.reduce_sum(loss), x_mu, z, z_mu, z_logvar, log_lik / self.batch_size

    def train_test(self):
        epochs, best_it = self.train()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        plot_cost(validation=self.validation_log_lik, training=self.train_log_lik, name='Log_lik', epochs=epochs,
                  best_iteration=best_it)
        plot_cost(validation=self.validation_cost, training=self.train_cost, name='Cost', epochs=epochs,
                  best_iteration=best_it)
        self.test_reconstruction()

    def test_reconstruction(self):
        # TODO improve reconstruction plot and plot many images
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
