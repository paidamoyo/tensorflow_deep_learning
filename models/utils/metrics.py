import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def convert_labels_to_cls(labels):
    return np.argmax(labels, axis=1)


def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


def plot_confusion_matrix(cls_pred, labels):
    cls_true = convert_labels_to_cls(labels)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)


def print_test_accuracy(correct, cls_pred, labels):
    acc, correct_sum = cls_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_images))

    print("Confusion Matrix:")
    plot_confusion_matrix(cls_pred=cls_pred, labels=labels)


def plot_images(x_test, x_reconstruct, n_images, name):
    assert len(x_test) == n_images
    print("x_reconstruct:{}, x_test:{}".format(x_reconstruct.shape, x_test.shape))

    plt.figure(figsize=(8, 12))
    for i in range(n_images):
        # Plot image.
        plt.subplot(n_images, 2, 2 * i + 1)
        s1 = plt.imshow(x_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.subplot(n_images, 2, 2 * i + 2)
        s2 = plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        s1.axes.get_xaxis().set_visible(False)
        s1.axes.get_yaxis().set_visible(False)
        s2.axes.get_xaxis().set_visible(False)
        s2.axes.get_yaxis().set_visible(False)

    # plt.title("Left: Test input and Right: Reconstruction")
    plt.tight_layout()
    save_path = name + "_reconstructed_digit"
    plt.savefig(save_path)
    # plt.axis('off')
