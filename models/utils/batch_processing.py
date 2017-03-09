def get_batch_size(num_examples, num_batches, num_lab):
    num_ulab = num_examples - num_lab
    assert num_lab % num_batches == 0, '#Labelled % #Batches != 0'
    assert num_ulab % num_batches == 0, '#Unlabelled % #Batches != 0'
    assert num_examples % num_batches == 0, '#Examples % #Batches != 0'

    batch_size = num_examples // num_batches
    num_lab_batch = num_lab // num_batches
    num_ulab_batch = num_ulab // num_batches
    print("num_lab_batch:{}, num_ulab_batch:{}, batch_size:{}".format(num_lab_batch, num_ulab_batch, batch_size))
    return num_lab_batch, num_ulab_batch, batch_size


def get_next_batch(x_images, y_labels, idx, batch_size):
    num_images = x_images.shape[0]
    if idx == num_images:
        idx = 0
    j = min(idx + batch_size, num_images)
    x_batch = x_images[idx:j, :]
    y_true_batch = y_labels[idx:j, :]
    return x_batch, y_true_batch, j


def get_encoded_next_batch(x_mu, x_logvar, y_labels, idx, batch_size):
    num_images = x_mu.shape[0]
    if idx == num_images:
        idx = 0
    j = min(idx + batch_size, num_images)
    x_mu_batch = x_mu[idx:j, :]
    x_logvar_batch = x_logvar[idx:j, :]
    y_true_batch = y_labels[idx:j, :]
    return x_mu_batch, x_logvar_batch, y_true_batch, j
