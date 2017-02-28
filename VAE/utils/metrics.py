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


def plot_images(x_test, x_reconstruct):
    assert len(x_test) == 10
    print("x_reconstruct:{}, x_test:{}".format(x_reconstruct.shape, x_test.shape))

    plt.figure(figsize=(8, 12))
    for i in range(10):
        # Plot image.
        plt.subplot(10, 2, 2 * i + 1)
        plt.imshow(x_test[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(10, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("reconstructed_digit")
