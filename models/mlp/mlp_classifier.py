import logging
import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from models.utils.MNIST_pickled_preprocess import extract_data
from models.utils.batch_processing import get_next_batch
from models.utils.metrics import convert_labels_to_cls, cls_accuracy, print_test_accuracy, plot_roc
from models.utils.tf_helpers import create_nn_weights, mlp_neuron


class MLPClassifier(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 input_dim,
                 num_classes,
                 hidden_dim=500
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.log_file = 'mlp_classifier.log'
        self.num_classes = num_classes
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ''' Create Graph '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/mlp_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):
        n_train_examples = 50000
        train_x_l, train_l_y, train_u_x, train_u_y, self.valid_x, self.valid_y, self.test_x, self.test_y = extract_data(
            n_train_examples)
        self.train_x = np.concatenate((train_x_l, train_u_x), axis=0)
        self.train_y = np.concatenate((train_l_y, train_u_y), axis=0)
        num_batches = int(n_train_examples / self.batch_size)
        logging.debug("num batches:{}, batch_size:{}, epochs:{}".format(num_batches, self.batch_size, int(
            self.num_iterations / num_batches)))

        self.y_logits, self.y_pred_cls, self.cost = self.build_model()
        tf.summary.scalar('cost', self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def train_neural_network(self):
        print_training = "Training MLP:"
        print(print_training)
        logging.debug(print_training)
        self.session.run(tf.global_variables_initializer())
        best_validation_accuracy = 0
        last_improvement = 0

        start_time = time.time()
        idx = 0

        for i in range(self.num_iterations):
            # Batch Training
            x_batch, y_batch, idx = get_next_batch(self.train_x, self.train_y, idx, self.batch_size)
            summary, batch_loss, _ = self.session.run([self.merged, self.cost, self.optimizer],
                                                      feed_dict={self.x: x_batch, self.y: y_batch})
            self.train_writer.add_summary(summary, i)

            if (i % 100 == 0) or (i == (self.num_iterations - 1)):
                # Calculate the accuracy
                correct, _ = self.predict_cls(images=self.valid_x,
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

                print_opt = "Iteration: {}, Training Loss: {}, " \
                            " Validation Acc:{}, {}".format(i + 1, batch_loss, acc_validation, improved_str)
                print(print_opt)
                logging.debug(print_opt)
            if i - last_improvement > self.require_improvement:
                print_impro = "No improvement found in a while, stopping optimization."
                print(print_impro)
                logging.debug(print_impro)
                # Break out from the for-loop.
                break
                # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(print_time)
        logging.debug(print_time)

    def predict_cls(self, images, labels, cls_true):
        num_images = len(images)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        i = 0
        mean_auc, batch_auc = tf.contrib.metrics.streaming_auc(predictions=self.y_logits, labels=self.y,
                                                               curve='ROC')
        self.session.run(tf.local_variables_initializer())
        final_mean_value = 0.0
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, num_images)
            batch_images = images[i:j, :]
            batch_labels = labels[i:j, :]
            feed_dict = {self.x: batch_images,
                         self.y: batch_labels}
            cls_pred[i:j] = self.session.run(self.y_pred_cls,
                                             feed_dict=feed_dict)
            i = j
            final_mean_value, auc = self.session.run([mean_auc, batch_auc], feed_dict=feed_dict)
        print_auc = 'Final Mean AUC: %f' % final_mean_value
        print(print_auc)
        logging.debug(print_auc)
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)
        return correct, cls_pred

    def build_model(self):
        with tf.variable_scope("y_classifier"):
            w_h1, b_h1 = create_nn_weights('y_h1', 'infer', [self.input_dim, self.hidden_dim])
            w_h2, b_h2 = create_nn_weights('y_h2', 'infer', [self.hidden_dim, self.hidden_dim])
            w_y, b_y = create_nn_weights('y_fully_connected', 'infer', [self.hidden_dim, self.num_classes])

            h1 = mlp_neuron(self.x, w_h1, b_h1)
            h2 = mlp_neuron(h1, w_h2, b_h2)
            logits = mlp_neuron(h2, w_y, b_y, activation=False)
            y_pred = tf.nn.softmax(logits)
            y_pred_cls = tf.argmax(y_pred, axis=1)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            cost = tf.reduce_mean(cross_entropy)
        return logits, y_pred_cls, cost

    def train_test(self):
        self.train_neural_network()
        self.saver.restore(sess=self.session, save_path=self.save_path)
        correct, cls_pred = self.predict_cls(images=self.test_x,
                                             labels=self.test_y,
                                             cls_true=(convert_labels_to_cls(self.test_y)))

        feed_dict = {self.x: self.test_x, self.y: self.test_y}
        logits = self.session.run(self.y_logits, feed_dict=feed_dict)
        plot_roc(logits, self.test_y, self.num_classes, name='MLP')
        print_test_accuracy(correct, cls_pred, self.test_y, logging)
