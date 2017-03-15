import matplotlib.pyplot as plt
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


def plot_images(x_test, x_reconstruct, n_images, name):
    assert len(x_test) == n_images
    print("x_reconstruct:{}, x_test:{}".format(x_reconstruct.shape, x_test.shape))

    plt.figure(figsize=(8, 12))
    for i in range(n_images):
        # Plot image.
        plt.subplot(n_images, 2, 2 * i + 1)
        s1 = plt.imshow(x_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.subplot(n_images, 2, 2 * i + 2)
        s2 = plt.imshow(x_reconstruct[i].reshape(4, 4), vmin=0, vmax=1, cmap="gray")
        s1.axes.get_xaxis().set_visible(False)
        s1.axes.get_yaxis().set_visible(False)
        s2.axes.get_xaxis().set_visible(False)
        s2.axes.get_yaxis().set_visible(False)

    # plt.title("Left: Test input and Right: Reconstruction")
    plt.tight_layout()
    save_path = name + "_reconstructed_digit"
    plt.savefig(save_path)
    # plt.axis('off')


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
        'n_components': 16
    }

    train_x_l, train_l_y, train_u_x, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(
        FLAGS['n_train'])
    train_x = np.concatenate((train_x_l, train_u_x), axis=0)
    train_y = np.concatenate((train_l_y, train_u_y), axis=0)

    components = pca_components(train_x, FLAGS['n_components'])
    train = [transform_inputs(components, train_x), train_y]
    valid = [transform_inputs(components, valid_x), valid_y]
    test = [transform_inputs(components, test_x), test_y]

    num_images = 20
    x_test = test_x[0:num_images, ]
    plot_images(x_test, transform_inputs(components, x_test), num_images, "pca")

    pca = PCAClassifier(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                        beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                        require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                        num_iterations=FLAGS['num_iterations'],
                        input_dim=FLAGS['n_components'], train=train, valid=valid, test=test,
                        num_classes=FLAGS['num_classes'])
    with pca.session:
        pca.train_test()
