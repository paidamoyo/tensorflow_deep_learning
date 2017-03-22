from models.auxiliary_semi_supervised.auxiliary_classifier import Auxiliary

if __name__ == '__main__':
    # Global Dictionary of Flags
    # learn_yz_x_ss.main(3000, n_labels=100, dataset='mnist_2layer', n_z=50, n_hidden=(300,), seed=seed, alpha=0.1,
    #                    n_minibatches=100, comment='')
    # learn_yz_x_ss.main(3000, n_labels>100, dataset='mnist_2layer', n_z=50, n_hidden=(500,), seed=seed, alpha=0.1,
    #                    n_minibatches=200, comment='')

    FLAGS = {
        'num_iterations': 300000,  # should 3000 epochs (30000/100)
        'batch_size': 200,
        'seed': 31415,
        'n_labeled': 100,
        'alpha': 0.1,
        'require_improvement': 5000,  # 50 epochs
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'num_classes': 10
    }

    aux = Auxiliary(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                    beta1=FLAGS['beta1'], beta2=FLAGS['beta2'], alpha=FLAGS['alpha'],
                    require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                    n_labeled=FLAGS['n_labeled'],
                    num_iterations=FLAGS['num_iterations'])  # Should be consistent with model being

    with aux.session:
        aux.train_test()
