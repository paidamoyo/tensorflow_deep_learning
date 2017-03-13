###
'''
Borrowed from original implementation: https://github.com/dpkingma/nips14-ssl (anglepy)
'''
###

import gzip
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


def load_numpy(path, binarize_y=False):
    # MNIST dataset
    f = gzip.open(path, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, valid, test = u.load()
    # train, valid, test = pickle.load(f)
    f.close()
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    if binarize_y:
        train_y = binarize_labels(train_y)
        valid_y = binarize_labels(valid_y)
        test_y = binarize_labels(test_y)

    return train_x.T, train_y, valid_x.T, valid_y, test_x.T, test_y


# Loads data where data is split into class labels
def load_numpy_split(binarize_y=False, n_train=50000):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '..', 'mnist/mnist_28.pkl.gz'))
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy(path, False)

    train_x = train_x[0:n_train]
    train_y = train_y[0:n_train]

    def split_by_class(x, y, num_classes):
        result_x = [0] * num_classes
        result_y = [0] * num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:, idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    train_x, train_y = split_by_class(train_x, train_y, 10)
    if binarize_y:
        valid_y = binarize_labels(valid_y)
        test_y = binarize_labels(test_y)
        for i in range(10):
            train_y[i] = binarize_labels(train_y[i])
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y


def unbinarize_labels(y):
    return np.argmax(y, axis=0)


def create_semisupervised(x, y, n_labeled):
    n_classes = y[0].shape[0]
    if n_labeled % n_classes != 0: raise (
        "n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = int(n_labeled / n_classes)
    print("n_labels_per_class {}".format(n_labels_per_class))
    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes

    for i in range(n_classes):
        num_train_per_class = x[i].shape[1]
        print(" class {}, num_train_per_class:{}".format(i, num_train_per_class))
        idx = np.arange(num_train_per_class)
        random.shuffle(idx)
        idx_labeled = idx[:n_labels_per_class]
        idx_unlabeled = idx[n_labels_per_class:]

        x_labeled[i] = x[i][:, idx_labeled]
        y_labeled[i] = y[i][:, idx_labeled]

        x_unlabeled[i] = x[i][:, idx_unlabeled]
        y_unlabeled[i] = y[i][:, idx_unlabeled]
    return np.hstack(x_labeled), np.hstack(y_labeled), np.hstack(x_unlabeled), np.hstack(y_unlabeled)


def binarize_images(images):
    bin_images = []
    for i in range(images.shape[0]):
        means = images[i]
        bin_images[i] = np.random.binomial(n=1, p=means)
    return bin_images


def extract_data(n_labeled):
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy_split(binarize_y=True)
    x_l, y_l, x_u, y_u = create_semisupervised(train_x, train_y, n_labeled)
    t_x_l, t_y_l = x_l.T, y_l.T
    t_x_u, t_y_u = x_u.T, y_u.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    print("x_l:{}, y_l:{}, x_u:{}, y_{}".format(t_x_l.shape, t_y_l.shape, t_x_u.shape, t_y_u.shape))
    return t_x_l, t_y_l, t_x_u, t_y_u, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':
    num_lab = 50000
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy_split(binarize_y=True)
    x_l, y_l, x_u, y_u = create_semisupervised(train_x, train_y, num_lab)

    x_lab, y_lab = x_l.T, y_l.T
    x_ulab, y_ulab = x_u.T, y_u.T
    print(x_lab.shape, y_lab.shape, x_ulab.shape, y_ulab.shape)
    print(x_lab[0])
    plt.imshow(x_lab[0].reshape(28, 28), cmap="gray")
    plt.show()
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T