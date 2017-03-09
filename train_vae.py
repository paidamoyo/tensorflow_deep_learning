from models.utils.MNIST_pickled_preprocess import extract_data
from models.vanilla_vae.vae import VariationalAutoencoder


def encode_dataset(FLAGS, train_lab, train_unlab, valid, test, train=True):
    vae = VariationalAutoencoder(batch_size=200, learning_rate=FLAGS['learning_rate'],
                                 beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                                 require_improvement=5000, seed=FLAGS['seed'],
                                 num_iterations=200,
                                 input_dim=FLAGS['input_dim'],
                                 latent_dim=FLAGS['latent_dim'])  # Should be consistent with model being loaded

    with vae.session:
        if train:
            vae.train_test()
        vae.saver.restore(vae.session, vae.save_path)

        enc_x_lab_mean, enc_x_lab_var = vae.encode(train_lab)
        enc_x_ulab_mean, enc_x_ulab_var = vae.encode(train_unlab)
        enc_x_valid_mean, enc_x_valid_var = vae.encode(valid)
        enc_x_test_mean, enc_x_test_var = vae.encode(test)

    return enc_x_lab_mean, enc_x_lab_var, enc_x_ulab_mean, enc_x_ulab_var, enc_x_valid_mean, \
           enc_x_valid_var, enc_x_test_mean, enc_x_test_var


if __name__ == '__main__':
    # Global Dictionary of Flags
    # learn_yz_x_ss.main(3000, n_labels=100, dataset='mnist_2layer', n_z=50, n_hidden=(300,), seed=seed, alpha=0.1,
    #                    n_minibatches=100, comment='')
    # learn_yz_x_ss.main(3000, n_labels>100, dataset='mnist_2layer', n_z=50, n_hidden=(500,), seed=seed, alpha=0.1,
    #                    n_minibatches=200, comment='')

    FLAGS = {
        'num_iterations': 200,
        'num_batches': 100,
        'seed': 12000,
        'n_labeled': 100,
        'alpha': 0.1,
        'latent_dim': 50,
        'require_improvement': 30000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10
    }

    train_x_lab, train_l_y, train_x_unlab, train_u_y, valid_x, valid_y, test_x, test_y = extract_data(
        FLAGS['n_labeled'])
    train_x_l_mu, train_x_l_logvar, train_x_u_mu, train_x_u_logvar, valid_x_mu, \
    valid_x_logvar, test_x_mu, test_x_logvar = encode_dataset(FLAGS=FLAGS, train_lab=train_x_lab,
                                                              train_unlab=train_x_unlab, valid=valid_x,
                                                              test=test_x)
