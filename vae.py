import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from MNSIT_prepocess import create_semisupervised
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('../')


def create_biases(shape):
    return tf.Variable(tf.random_normal(shape))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def variable_summaries(var, summary_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


## Build Model
def recognition_network():
    # Variables
    W_encoder_h_1, b_encoder_h_1 = create_h_weights('h1', 'encoder', [img_size_flat, encoder_h_dim])
    W_encoder_h_2, b_encoder_h_2 = create_h_weights('h2', 'encoder', [encoder_h_dim, encoder_h_dim])
    W_enconder_h_mu_z1, W_enconder_h_var_z1, b_enconder_h_mu_z1, b_enconder_h_var_z1 = create_z_weights('z_1',
                                                                                                        [encoder_h_dim,

                                                                                                         latent_dim])
    W_encoder_h_3, b_encoder_h_3 = create_h_weights('h3', 'encoder', [latent_dim, encoder_h_dim])
    W_encoder_h_4, b_encoder_h_4 = create_h_weights('h4', 'encoder', [encoder_h_dim, encoder_h_dim])

    W_enconder_h_mu_z2, W_enconder_h_var_z2, b_enconder_h_mu_z2, b_enconder_h_var_z2 = create_z_weights('z_2',
                                                                                                        [encoder_h_dim,
                                                                                                         latent_dim])

    # Model

    # Hidden layers
    encoder_h_1 = activated_neuron(x, W_encoder_h_1, b_encoder_h_1)
    encoder_h_2 = activated_neuron(encoder_h_1, W_encoder_h_2, b_encoder_h_2)

    # Z1 latent layer mu and var
    encoder_logvar_z1 = activated_neuron(encoder_h_2, W_enconder_h_var_z1, b_enconder_h_var_z1)
    encoder_mu_z1 = non_activated_neuron(encoder_h_2, W_enconder_h_mu_z1, b_enconder_h_mu_z1)
    z_1 = draw_z(latent_dim, encoder_mu_z1, encoder_logvar_z1)

    # Hidden layers
    encoder_h_3 = activated_neuron(z_1, W_encoder_h_3, b_encoder_h_3)
    encoder_h_4 = activated_neuron(encoder_h_3, W_encoder_h_4, b_encoder_h_4)

    # Z2 latent layer mu and var
    encoder_logvar_z2 = activated_neuron(encoder_h_4, W_enconder_h_var_z2, b_enconder_h_var_z2)
    encoder_mu_z2 = non_activated_neuron(encoder_h_4, W_enconder_h_mu_z2, b_enconder_h_mu_z2)
    z_2 = draw_z(latent_dim, encoder_mu_z2, encoder_logvar_z2)

    # regularization loss
    regularization = -0.5 * tf.reduce_sum(1 + encoder_logvar_z2 - tf.pow(encoder_mu_z2, 2) - tf.exp(encoder_logvar_z2),
                                          reduction_indices=1)

    return z_2, regularization


def draw_z(dim, mu, logvar):
    epsilon_encoder = tf.random_normal(tf.shape(dim), name='epsilon')
    std_encoder_z1 = tf.exp(0.5 * logvar)
    z = mu + tf.mul(std_encoder_z1, epsilon_encoder)
    return z


def non_activated_neuron(input, Weights, biases):
    return tf.add(tf.matmul(input, Weights), biases)


def activated_neuron(input, Weights, biases):
    return tf.nn.relu(tf.add(tf.matmul(input, Weights), biases))


def create_h_weights(layer, network, shape):
    h_vars = {}
    W_h = 'W_' + network + '_' + layer
    b_h = 'b_' + network + '_' + layer
    print("layer:{}, network:{}, shape:{}".format(layer, network, shape))
    h_vars[W_h] = create_weights(shape)
    h_vars[b_h] = create_biases([shape[1]])
    variable_summaries(h_vars[W_h], W_h)
    variable_summaries(h_vars[b_h], b_h)

    return h_vars[W_h], h_vars[b_h]


def create_z_weights(layer, shape):
    print("layer:{}, z_latent, shape:{}".format(layer, shape))
    z_vars = {}
    network = 'encoder'

    # Mean
    W_z_mu = 'W_' + network + '_h_mu_' + layer
    b_z_mu = 'b_' + network + '_h_mu_' + layer
    z_vars[W_z_mu] = create_weights(shape)
    z_vars[b_z_mu] = create_biases([shape[1]])
    variable_summaries(z_vars[W_z_mu], W_z_mu)
    variable_summaries(z_vars[b_z_mu], b_z_mu)

    # Variance
    W_z_var = 'W_' + network + '_h_var_' + layer
    b_z_var = 'b_' + network + '_h_var_' + layer
    z_vars[W_z_var] = create_weights(shape)
    z_vars[b_z_var] = create_biases([shape[1]])
    variable_summaries(z_vars[W_z_var], W_z_var)
    variable_summaries(z_vars[b_z_var], b_z_var)

    return z_vars[W_z_mu], z_vars[W_z_var], z_vars[b_z_mu], z_vars[b_z_var]


def generator_network():
    # Variables
    W_decoder_h_1, b_decoder_h_1 = create_h_weights('h1', 'decoder', [latent_dim, decoder_h_dim])
    W_decoder_h_2, b_decoder_h_2 = create_h_weights('h2', 'decoder', [decoder_h_dim, decoder_h_dim])
    W_decoder_r, b_decoder_r = create_h_weights('r', 'decoder', [decoder_h_dim, img_size_flat])

    # Model
    # Decoder hidden layer
    decoder_h_1 = activated_neuron(z, W_decoder_h_1, b_decoder_h_1)
    decoder_h_2 = activated_neuron(decoder_h_1, W_decoder_h_2, b_decoder_h_2)

    # Reconstrunction layer
    x_hat = non_activated_neuron(decoder_h_2, W_decoder_r, b_decoder_r)
    tf.summary.image('x_hat', tf.reshape(x_hat[0], [1, 28, 28, 1]))
    return x_hat


def train_neural_network(num_iterations, loss_function, input_data, init=False):
    optimizer = tf.train.AdamOptimizer().minimize(loss_function)
    session.run(tf.global_variables_initializer())
    # Ensure we update the global variable rather than a local copy.
    total_iterations = 0

    start_time = time.time()
    num_images, x_images, y_labels = preprocess_train_data(input_data)

    i = 0

    for epoch in range(num_iterations):
        if i == num_images:
            i = 0
        # The ending index for the next batch is denoted j.
        j = min(i + FLAGS['train_batch_size'], num_images)

        # Get the images from the test-set between index i and j.
        x_batch = x_images[i:j, :]

        # Get the associated labels.
        y_true_batch = y_labels[i:j, :]

        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        summary, batch_loss, _ = session.run([merged, loss_function, optimizer], feed_dict=feed_dict_train)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        train_writer.add_summary(summary, batch_loss)
        i = j

        if epoch % 100 == 0:
            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=session, save_path=FLAGS['save_path'])
            print("Optimization Iteration: {}, Training Loss: {}".format(epoch + 1, batch_loss))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def preprocess_train_data(input_data):
    x_images = input_data[0]
    y_labels = input_data[1]
    num_images = x_images.shape[0]
    randomize = np.arange(num_images)
    np.random.shuffle(randomize)
    x_images = x_images[randomize]
    y_labels = y_labels[randomize]
    return num_images, x_images, y_labels


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


def mlp_classifier(latent_x):
    W_mlp_h1 = create_weights([latent_dim, num_classes])
    b_mlp_h1 = create_biases([num_classes])

    logits = tf.matmul(latent_x, W_mlp_h1) + b_mlp_h1
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true)
    return cross_entropy, y_pred_cls


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
    }

    np.random.seed(FLAGS['seed'])
    data = input_data.read_data_sets(FLAGS['data_directory'], one_hot=True)

    encoder_h_dim = 500
    decoder_h_dim = 500
    latent_dim = 50
    img_size = 28
    num_classes = 10
    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size
    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # ### Placeholder variables
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Encoder Model
    z, recognition_loss = recognition_network()

    # MLP Classification Network
    cross_entropy_loss, y_pred_cls = mlp_classifier(z)

    alpha = 0.1

    # Decoder Model
    x_hat = generator_network()

    reconstruction_loss = tf.reduce_sum(tf.squared_difference(x_hat, x), reduction_indices=1)

    labeled_loss = tf.reduce_mean(
        recognition_loss + reconstruction_loss + alpha * FLAGS['train_batch_size'] * cross_entropy_loss)
    tf.summary.scalar('labeled_loss', labeled_loss)

    unlabeled = tf.reduce_mean(
        recognition_loss + reconstruction_loss)
    tf.summary.scalar('unlabeled_loss', unlabeled)

    session = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS['summaries_dir'] + '/train',
                                         session.graph)

    # Counter for total number of iterations performed so far.
    total_iterations = 0

    # create labeled/unlabeled split in training set
    n_labeled = 10000
    x_l, y_l, x_u, y_u = create_semisupervised(n_labeled)
    print("x_l:{}, y_l:{}, x_u:{}, y_{}".format(x_l.shape, y_l.shape, x_u.shape, y_u.shape))

    ## SAVER
    saver = tf.train.Saver()
    # train labeled
    print("labeled training:")
    train_neural_network(FLAGS['num_iterations'], loss_function=labeled_loss, input_data=[x_l, y_l], init=True)
    print("unlabeled training:")
    train_neural_network(FLAGS['num_iterations'], loss_function=unlabeled, input_data=[x_u, y_u])
    # train unlabeled
    test_reconstruction()
    print_test_accuracy()

    session.close()
