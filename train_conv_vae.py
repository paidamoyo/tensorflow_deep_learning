import sys

from models.conv_vae.convolutional_vae import ConvVariationalAutoencoder
from models.utils.MNIST_pickled_preprocess import extract_data


def encode_dataset(FLAGS, train_lab, train_unlab, valid, test, gpu_memory_fraction, train=True):
    conv_vae = ConvVariationalAutoencoder(batch_size=250, learning_rate=FLAGS['learning_rate'],
                                          beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                                          require_improvement=1000, seed=FLAGS['seed'],
                                          num_iterations=FLAGS['num_iterations'],
                                          input_dim=FLAGS['input_dim'],
                                          latent_dim=FLAGS['latent_dim'],
                                          filter_sizes=FLAGS['filter_sizes'], fc_size=FLAGS['fc_size'],
                                          num_filters=FLAGS[
                                              'num_filters'],
                                          gpu_memory_fraction=gpu_memory_fraction)  # Should be consistent with model being loaded

    with conv_vae.session:
        if train:
            conv_vae.train_test()
        conv_vae.saver.restore(conv_vae.session, conv_vae.save_path)

        enc_x_lab_mean, enc_x_lab_var = conv_vae.encode(train_lab)
        enc_x_ulab_mean, enc_x_ulab_var = conv_vae.encode(train_unlab)
        enc_x_valid_mean, enc_x_valid_var = conv_vae.encode(valid)
        enc_x_test_mean, enc_x_test_var = conv_vae.encode(test)

    return enc_x_lab_mean, enc_x_lab_var, enc_x_ulab_mean, enc_x_ulab_var, enc_x_valid_mean, \
           enc_x_valid_var, enc_x_test_mean, enc_x_test_var


if __name__ == '__main__':
    FLAGS_ = {
        'num_iterations': 40000,  # should 3000 epochs
        'num_batches': 100,
        'seed': 31415,
        'n_labeled': 50000,
        'alpha': 0.1,
        'latent_dim': 50,
        'require_improvement': 5000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10,
        'min_std': 0.1,  # Dimensions with std < min_std are removed before training with GC
        'l2_weight': 1e-6,
        'filter_sizes': [5, 5],
        'num_filters': [16, 36],
        'fc_size': 128
    }
    args = sys.argv[1:]
    print("args:{}".format(args))
    if args:
        vm = float(args[0])
    else:
        vm = 1.0
    print("gpu_memory_fraction:{}".format(vm))
    train_x_lab, train_l_y, train_x_unlab, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(
        FLAGS_['n_labeled'])

    train_x_l_mu, train_x_l_logvar, train_x_u_mu, train_x_u_logvar, valid_x_mu, \
 \
    valid_x_logvar, test_x_mu, test_x_logvar = encode_dataset(FLAGS=FLAGS_, train_lab=train_x_lab,
                                                              train_unlab=train_x_unlab, valid=valid_x,
                                                              test=test_x, gpu_memory_fraction=vm)
