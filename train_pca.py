import numpy as np
from sklearn.decomposition import PCA

from models.pca.pca_classifier import PCAClassifier
from models.utils.MNIST_pickled_preprocess import extract_data


def pca_components(data, n_components):
    mean = np.mean(data, 0)
    constant = 1e-10
    std = np.std(data, 0) + constant
    print_moments = "mean {}, std:{}".format(mean.shape, std.shape)
    print(print_moments)
    normalized_x = (data - mean) / std  # You need to normalize your data first
    print(normalized_x.shape)

    pca = PCA(n_components=n_components).fit(
        normalized_x)  # n_components is the components number after reduction
    print_components = "components:{}".format(pca.components_.shape)
    print(print_components)
    return pca.components_


def transform_inputs(components, data):
    return np.dot(data, components.T)


if __name__ == '__main__':
    FLAGS = {
        'num_iterations': 40000,  # should 3000 epochs
        'batch_size': 200,
        'seed': 31415,
        'alpha': 0.1,
        'require_improvement': 5000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'num_classes': 10,
        'n_components': 22
    }

    train_x_l, train_l_y, train_u_x, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(
        FLAGS['n_train'])
    train_x = np.concatenate((train_x_l, train_u_x), axis=0)
    train_y = np.concatenate((train_l_y, train_u_y), axis=0)

    components = pca_components(train_x, FLAGS['n_components'])
    train = [transform_inputs(components, train_x), train_y]
    valid = [transform_inputs(components, valid_x), valid_y]
    test = [transform_inputs(components, test_x), test_y]

    pca = PCAClassifier(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                        beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                        require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                        num_iterations=FLAGS['num_iterations'],
                        input_dim=FLAGS['n_components'], train=train, valid=valid, test=test,
                        num_classes=FLAGS['num_classes'])
    with pca.session:
        pca.train_test()
