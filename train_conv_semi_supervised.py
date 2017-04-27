import sys

from models.semi_supervised_conv_vae.conv_semi_supervised import ConvGenerativeClassifier
from models.utils.MNIST_pickled_preprocess import extract_data
from train_conv_vae import encode_dataset

if __name__ == '__main__':
    # Global Dictionary of Flags
    # learn_yz_x_ss.main(3000, n_labels=100, dataset='mnist_2layer', n_z=50, n_hidden=(300,), seed=seed, alpha=0.1,
    #                    n_minibatches=100, comment='')
    # learn_yz_x_ss.main(3000, n_labels>100, dataset='mnist_2layer', n_z=50, n_hidden=(500,), seed=seed, alpha=0.1,
    #                    n_minibatches=200, comment='')

    FLAGS = {
        'num_iterations': 1,  # should 3000 epochs
        'num_batches': 250,
        'seed': 31415,
        'n_labeled': 100,  # should be 100
        'alpha': 0.1,  # 0.1 - 2 TODO change alpha back to 0.1
        'latent_dim': 50,  # should be 50  TODO change back to 50
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
        FLAGS['n_labeled'])
    train_x_l_mu, train_x_l_logvar, train_x_u_mu, train_x_u_logvar, valid_x_mu, \
    valid_x_logvar, test_x_mu, test_x_logvar = encode_dataset(FLAGS=FLAGS, train_lab=train_x_lab,
                                                              train_unlab=train_x_unlab, valid=valid_x,
                                                              test=test_x, gpu_memory_fraction=vm)
    train_lab = [train_x_l_mu, train_x_l_logvar, train_l_y]
    train_unlab = [train_x_u_mu, train_x_u_logvar, train_u_y]
    valid = [valid_x_mu, valid_x_logvar, valid_y]
    test = [test_x_mu, test_x_logvar, test_y]

    print("train lab: mu {}, var:{}, y:{}".format(train_x_l_mu.shape, train_x_l_logvar.shape, train_l_y.shape))
    print("train unlab: mu {}, var:{}, y:{}".format(train_x_u_mu.shape, train_x_u_logvar.shape, train_u_y.shape))
    print("valid: mu {}, var:{}, y:{}".format(valid_x_mu.shape, valid_x_logvar.shape, valid_y.shape))
    print("test: mu {}, var:{}, y:{}".format(test_x_mu.shape, test_x_logvar.shape, test_y.shape))

    genclass = ConvGenerativeClassifier(num_batches=FLAGS['num_batches'], learning_rate=FLAGS['learning_rate'],
                                        beta1=FLAGS['beta1'], beta2=FLAGS['beta2'], alpha=FLAGS['alpha'],
                                        require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                                        n_labeled=FLAGS['n_labeled'],
                                        num_iterations=FLAGS['num_iterations'],
                                        input_dim=train_x_l_mu.shape[1],
                                        latent_dim=FLAGS['latent_dim'],
                                        train_lab=train_lab, train_unlab=train_unlab, valid=valid,
                                        test=test, filter_sizes=FLAGS['filter_sizes'], fc_size=FLAGS['fc_size'],
                                        num_filters=FLAGS[
                                            'num_filters'],
                                        gpu_memory_fraction=vm)  # Should be consistent with model being
    with genclass.session:
        genclass.train_test()
