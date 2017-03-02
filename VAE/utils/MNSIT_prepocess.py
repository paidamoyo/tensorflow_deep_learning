import numpy as np


def split_by_class(data, n_train):
    train_x = data.train.images[0:n_train]
    train_y = data.train.labels[0:n_train]
    n_classes = train_y.shape[1]
    y_cls = np.argmax(train_y, axis=1)
    result_x = [0] * n_classes
    result_y = [0] * n_classes
    for i in range(n_classes):
        idx_i = np.where(y_cls == i)[0]
        result_x[i] = train_x[idx_i]
        result_y[i] = train_y[idx_i]
    return result_x, result_y


def split_data(data, n_labeled, n_train):
    train_x, train_y = split_by_class(data, n_train)
    n_classes = train_y[0].shape[1]
    print("n_classes:{}".format(n_classes))

    if n_labeled % n_classes != 0: raise (
        "n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = int(n_labeled / n_classes)
    print("n_labels_per_class:{}".format(n_labels_per_class))
    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    for i in range(n_classes):
        idx = np.arange(train_x[i].shape[0])
        np.random.shuffle(idx)
        label_idx = idx[:n_labels_per_class]
        unlabeled_idx = idx[n_labels_per_class:]

        x_labeled[i] = train_x[i][label_idx]
        y_labeled[i] = train_y[i][label_idx]

        x_unlabeled[i] = train_x[i][unlabeled_idx]
        y_unlabeled[i] = train_y[i][unlabeled_idx]

    return np.vstack(x_labeled), np.vstack(y_labeled), np.vstack(x_unlabeled), np.vstack(y_unlabeled)


def preprocess_train_data(data, n_labeled, n_train):
    # create labeled/unlabeled split in training set
    x_l, y_l, x_u, y_u = split_data(data=data, n_labeled=n_labeled, n_train=n_train)
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
