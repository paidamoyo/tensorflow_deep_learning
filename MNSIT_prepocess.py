import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_numpy():
    data = input_data.read_data_sets('data/MNIST/', one_hot=True)
    train_images = data.train.images
    train_labels = data.train.labels
    val_images = data.validation.images
    val_labels = data.validation.labels
    test_images = data.test.images
    test_labels = data.test.labels

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def load_numpy_split(n_train=50000):
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy()

    train_x = train_x[0:n_train]
    train_y = train_y[0:n_train]

    def split_by_class(x, y, num_classes):
        print("train_x:{}, train_y:{}".format(train_x.shape, train_y.shape))
        y_cls = np.argmax(y, axis=1)
        result_x = [0] * num_classes
        result_y = [0] * num_classes
        for i in range(num_classes):
            idx_i = np.where(y_cls == i)[0]
            print("y_cls:{}, class:{}, index_data:{}".format(y_cls, i, idx_i))
            result_x[i] = x[idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y

    train_x, train_y = split_by_class(train_x, train_y, 10)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def create_semisupervised(n_labeled):
    x, y, _, _, _, _ = load_numpy_split()
    print("x:{}, y:{}".format(x[0].shape, y[0].shape))
    n_x = x[0].shape[0]
    n_classes = y[0].shape[1]
    print("n_classes:{}".format(n_classes))

    if n_labeled % n_classes != 0: raise (
        "n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = n_labeled / n_classes
    print("n_labels_per_class:{}".format(n_labels_per_class))
    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    for i in range(n_classes):
        idx = np.arange(x[i].shape[0])
        print("idx:{}".format(idx))
        np.random.shuffle(idx)
        print("idx shuffled:{}".format(idx))
        label_idx = idx[:n_labels_per_class]
        unlabeled_idx = idx[n_labels_per_class:]

        x_labeled[i] = x[i][label_idx]
        y_labeled[i] = y[i][label_idx]

        x_unlabeled[i] = x[i][unlabeled_idx]
        y_unlabeled[i] = y[i][unlabeled_idx]

    return np.vstack(x_labeled), np.vstack(y_labeled), np.vstack(x_unlabeled), np.vstack(y_unlabeled)


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy_split()
    create_semisupervised(10000)
