import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_helper as helper
from MNSIT_prepocess import create_semisupervised
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('../')


def generate_z1():
    # Variables
    w_encoder_h_1, b_encoder_h_1 = helper.create_h_weights('h1', 'encoder', [img_size_flat, FLAGS['encoder_h_dim']])
    w_encoder_h_2, b_encoder_h_2 = helper.create_h_weights('h2', 'encoder',
                                                           [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])
    w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = helper.create_z_weights('z_1', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Hidden layers
    encoder_h_1 = helper.activated_neuron(x, w_encoder_h_1, b_encoder_h_1)
    encoder_h_2 = helper.activated_neuron(encoder_h_1, w_encoder_h_2, b_encoder_h_2)

    # Z1 latent layer mu and var
    encoder_logvar_z1 = helper.non_activated_neuron(encoder_h_2, w_var_z1, b_var_z1)
    encoder_mu_z1 = helper.non_activated_neuron(encoder_h_2, w_mu_z1, b_mu_z1)
    return draw_z(FLAGS['latent_dim'], encoder_mu_z1, encoder_logvar_z1)


# Build Model
def recognition_network():
    global y_logits
    # Variables
    w_encoder_h_3, b_encoder_h_3 = helper.create_h_weights('h3', 'encoder',
                                                           [FLAGS['latent_dim'], FLAGS['encoder_h_dim']])
    w_encoder_h_4, b_encoder_h_4 = helper.create_h_weights('h4', 'encoder',
                                                           [FLAGS['encoder_h_dim'], FLAGS['encoder_h_dim']])
    w_encoder_h_4_mu, b_encoder_h_4_mu = helper.create_h_weights('h4_mu', 'encoder', [FLAGS['encoder_h_dim'],
                                                                                      FLAGS[
                                                                                          'encoder_h_dim'] - num_classes])

    w_mu_z2, w_var_z2, b_mu_z2, b_var_z2 = helper.create_z_weights('z_2', [FLAGS['encoder_h_dim'], FLAGS['latent_dim']])

    # Model
    z_1 = generate_z1()
    # Hidden layers
    encoder_h_3 = helper.activated_neuron(z_1, w_encoder_h_3, b_encoder_h_3)
    encoder_h_4 = helper.activated_neuron(encoder_h_3, w_encoder_h_4, b_encoder_h_4)
    encoder_h_4_mu = helper.activated_neuron(encoder_h_3, w_encoder_h_4_mu, b_encoder_h_4_mu)

    # Z2 latent layer mu and var
    y_logits = predict_y(z_1)
    encoder_logvar_z2 = helper.non_activated_neuron(encoder_h_4, w_var_z2, b_var_z2)
    encoder_mu_z2 = helper.non_activated_neuron(tf.concat((y_logits, encoder_h_4_mu), axis=1), w_mu_z2,
                                                b_mu_z2)
    z_2 = draw_z(FLAGS['latent_dim'], encoder_mu_z2, encoder_logvar_z2)

    # regularization loss
    regularization = calculate_regularization_loss(encoder_logvar_z2, encoder_mu_z2)

    return z_2, regularization


def calculate_regularization_loss(encoder_logvar_z2, encoder_mu_z2):
    return -0.5 * tf.reduce_sum(1 + encoder_logvar_z2 - tf.pow(encoder_mu_z2, 2) - tf.exp(encoder_logvar_z2),
                                axis=1)


def draw_z(dim, mu, logvar):
    epsilon_encoder = tf.random_normal(tf.shape(dim), name='epsilon')
    std_encoder_z1 = tf.exp(0.5 * logvar)
    return mu + tf.multiply(std_encoder_z1, epsilon_encoder)


def generator_network():
    # Variables
    w_decoder_h_3, b_decoder_h_3 = helper.create_h_weights('h3', 'decoder',
                                                           [FLAGS['latent_dim'], FLAGS['decoder_h_dim']])
    w_decoder_h_4, b_decoder_h_4 = helper.create_h_weights('h4', 'decoder',
                                                           [FLAGS['decoder_h_dim'], FLAGS['decoder_h_dim']])
    w_decoder_mu, b_decoder_mu = helper.create_h_weights('mu', 'decoder', [FLAGS['decoder_h_dim'], img_size_flat])
    # w_decoder_var, b_decoder_var = create_h_weights('var', 'decoder', [FLAGS['decoder_h_dim'], img_size_flat])
    # Model
    # Decoder hidden layer
    decoder_h_3 = helper.activated_neuron(decoder_z1(), w_decoder_h_3, b_decoder_h_3)
    decoder_h_4 = helper.activated_neuron(decoder_h_3, w_decoder_h_4, b_decoder_h_4)

    # Reconstruction layer
    x_mu = helper.non_activated_neuron(decoder_h_4, w_decoder_mu, b_decoder_mu)
    # x_logvar = non_activated_neuron(decoder_h_2, w_decoder_var, b_decoder_var)
    tf.summary.image('x_mu', tf.reshape(x_mu[0], [1, 28, 28, 1]))
    return x_mu


def decoder_z1():
    w_decoder_h_1, b_decoder_h_1 = helper.create_h_weights('h1', 'decoder',
                                                           [FLAGS['latent_dim'] + num_classes, FLAGS['decoder_h_dim']])
    w_decoder_h_2, b_decoder_h_2 = helper.create_h_weights('h2', 'decoder',
                                                           [FLAGS['decoder_h_dim'], FLAGS['decoder_h_dim']])

    w_mu_z1, w_var_z1, b_mu_z1, b_var_z1 = helper.create_z_weights('z_1_decoder',
                                                                   [FLAGS['decoder_h_dim'], FLAGS['latent_dim']])
    # Model
    # Decoder hidden layer
    decoder_h_1 = helper.activated_neuron(tf.concat((y_logits, z_latent_rep), axis=1), w_decoder_h_1, b_decoder_h_1)
    decoder_h_2 = helper.activated_neuron(decoder_h_1, w_decoder_h_2, b_decoder_h_2)

    # Z1 latent layer mu and var
    decoder_logvar_z1 = helper.non_activated_neuron(decoder_h_2, w_var_z1, b_var_z1)
    decoder_mu_z1 = helper.non_activated_neuron(decoder_h_2, w_mu_z1, b_mu_z1)
    return draw_z(FLAGS['latent_dim'], decoder_mu_z1, decoder_logvar_z1)


def train_neural_network(num_iterations):
    session.run(tf.global_variables_initializer())
    best_validation_accuracy = 0
    last_improvement = 0

    start_time = time.time()
    x_l, y_l, x_u, y_u = preprocess_train_data()

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
            acc_validation, _ = validation_accuracy()
            if acc_validation > best_validation_accuracy:
                # Save  Best Perfoming all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=FLAGS['save_path'])
                # update best validation accuracy
                best_validation_accuracy = acc_validation
                last_improvement = epoch
                improved_str = '*'
            else:
                improved_str = ''

            print("Optimization Iteration: {}, {} Training Loss: {},  Validation Acc:{}, {}".format(epoch + 1,
                                                                                                    loss_string,
                                                                                                    batch_loss,
                                                                                                    acc_validation,
                                                                                                    improved_str))
        if epoch - last_improvement > FLAGS['require_improvement']:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def train_batch(idx, x_images, y_labels, loss, optimizer):
    # Batch Training
    num_images = x_images.shape[0]
    if idx == num_images:
        idx = 0
        # The ending index for the next batch is denoted j.
    j = min(idx + FLAGS['train_batch_size'], num_images)
    # Get the mages from the test-set between index idx_labeled and j.
    x_batch = x_images[idx:j, :]
    # Get the associated labels.
    y_true_batch = y_labels[idx:j, :]
    feed_dict_train = {x: x_batch, y_true: y_true_batch}
    summary, batch_loss, _ = session.run([merged, loss, optimizer], feed_dict=feed_dict_train)
    # Set the start-index for the next batch to the
    # end-index of the current batch.
    train_writer.add_summary(summary, batch_loss)
    return batch_loss, j


def preprocess_train_data():
    # create labeled/unlabeled split in training set
    n_labeled = FLAGS['n_labeled']
    x_l, y_l, x_u, y_u = create_semisupervised(n_labeled)
    print("x_l:{}, y_l:{}, x_u:{}, y_{}".format(x_l.shape, y_l.shape, x_u.shape, y_u.shape))
    # Labeled
    num_l = x_l.shape[0]
    randomize_l = np.arange(num_l)
    np.random.shuffle(randomize_l)
    x_l = x_l[randomize_l]
    y_l = y_l[randomize_l]

    # Unlabeled
    num_u = x_u.shape[0]
    randomize_u = np.arange(num_u)
    x_u = x_u[randomize_u]
    y_u = y_u[randomize_u]

    return x_l, y_l, x_u, y_u


def reconstruct(x_test):
    return session.run(x_hat, feed_dict={x: x_test})


def plot_images(x_test, x_reconstruct):
    assert len(x_test) == 5

    plt.figure(figsize=(8, 12))
    for i in range(5):
        # Plot image.
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("reconstructed_digit")


def test_reconstruction():
    saver.restore(sess=session, save_path=FLAGS['save_path'])
    x_test = data.test.next_batch(100)[0][0:5, ]
    print(np.shape(x_test))
    x_reconstruct = reconstruct(x_test)
    plot_images(x_test, x_reconstruct)


def mlp_classifier():
    global y_pred_cls
    y_pred = tf.nn.softmax(y_logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_true)
    return cross_entropy, y_pred_cls


def predict_y(z_1):
    w_mlp_h1, b_mlp_h1 = helper.create_h_weights('mlp_h1', 'classifier', [FLAGS['latent_dim'], FLAGS['latent_dim']])
    w_mlp_h2, b_mlp_h2 = helper.create_h_weights('mlp_h2', 'classifier', [FLAGS['latent_dim'], num_classes])

    h1 = helper.activated_neuron(z_1, w_mlp_h1, b_mlp_h1)
    logits = helper.non_activated_neuron(h1, w_mlp_h2, b_mlp_h2)
    return logits


def predict_cls(images, labels, cls_true):
    num_images = len(images)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['test_batch_size'], num_images)

        # Get the images from the test-set between index i and j.
        test_images = images[i:j, :]

        # Get the associated labels.
        labels = labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: test_images,
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def convert_labels_to_cls(labels):
    return np.argmax(labels, axis=1)


def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()
    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


def plot_confusion_matrix(cls_pred):
    cls_true = convert_labels_to_cls(data.test.labels)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)
    # Plot the confusion matrix as an image.
    plt.matshow(cm)


def validation_accuracy():
    correct, _ = predict_cls(images=data.validation.images,
                             labels=data.validation.labels,
                             cls_true=convert_labels_to_cls(data.validation.labels))
    return cls_accuracy(correct)


def print_test_accuracy():
    correct, cls_pred = predict_cls(images=data.test.images,
                                    labels=data.test.labels,
                                    cls_true=(convert_labels_to_cls(data.test.labels)))
    acc, correct_sum = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_images))

    print("Confusion Matrix:")
    plot_confusion_matrix(cls_pred=cls_pred)


def compute_labeled_loss():
    # === Get gradient for labeled data
    # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
    # logpx, logpz, logqz, gv_labeled, gw_labeled
    # _L = log_prior_y + log_lik + log_prior_z - log_post_z
    # Reweight gu_labeled and logqy
    # beta = FLAGS['alpha'] * (1.0 * FLAGS['train_batch_size'] / FLAGS['n_labeled'])
    beta = FLAGS['alpha'] * (1.0 * FLAGS['n_labeled'])
    cross_entropy_loss, _ = mlp_classifier()
    weighted_classification_loss = beta * cross_entropy_loss
    loss = tf.reduce_mean(
        recognition_loss + reconstruction_loss() + weighted_classification_loss)
    tf.summary.scalar('labeled_loss', loss)

    return loss


def compute_unlabeled_loss():
    # === Get gradient for unlabeled data
    # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
    # Approach where outer expectation (over q(z|x,y)) is taken as explicit sum (instead of sampling)
    # logpx, logpz, logqz, _gv, _gw = model.dL_dw(v, w, {'x':x_minibatch['x'],'y':new_y}, eps)
    # Decoder Model
    pi = tf.nn.softmax(y_logits)
    entropy = tf.einsum('ij,ij->i', pi, tf.log(pi))
    vae_loss = recognition_loss + reconstruction_loss()
    weighted_loss = tf.einsum('ij,ik->i', tf.reshape(vae_loss, [FLAGS['train_batch_size'], 1]), pi)
    print("entropy:{}, pi:{}, weighted_loss:{}".format(entropy, pi, weighted_loss))
    loss = tf.reduce_mean(weighted_loss)
    tf.summary.scalar('unlabeled_loss', loss)
    return loss


def reconstruction_loss():
    # exp_term = tf.squared_difference(x_hat, x) / (2.0 * tf.exp(x_logvar))
    # reconstruction_1D = tf.add(0.5 * x_logvar, exp_term)
    return tf.reduce_sum(tf.squared_difference(x_hat, x), 1)


if __name__ == '__main__':
    # Global Dictionary of Flags
    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'train_batch_size': 100,
        'test_batch_size': 256,
        'num_iterations': 10000,
        'seed': 12000,
        'n_labeled': 3000,
        'alpha': 0.1,
        'encoder_h_dim': 500,
        'decoder_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 1500,
        'n_total': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999
    }

    np.random.seed(FLAGS['seed'])
    data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

    img_size = 28
    num_classes = 10
    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size
    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # ### Placeholder variables
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # Encoder Model
    z_latent_rep, recognition_loss = recognition_network()
    # Decoder Model
    x_hat = generator_network()
    # MLP Classification Network

    labeled_loss = compute_labeled_loss()

    unlabeled_loss = compute_unlabeled_loss()

    session = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                         session.graph)

    labeled_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                               beta2=FLAGS['beta2']).minimize(
        labeled_loss)
    unlabeled_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                                                 beta2=FLAGS['beta2']).minimize(
        unlabeled_loss)

    saver = tf.train.Saver()

    train_neural_network(FLAGS['num_iterations'])
    print_test_accuracy()
    test_reconstruction()

    session.close()
