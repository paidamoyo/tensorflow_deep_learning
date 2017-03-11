from models.semi_supervised_vae.pretrained_semi_supervised import PreTrainedGenerativeClassifier

if __name__ == '__main__':
    # Global Dictionary of Flags
    # learn_yz_x_ss.main(3000, n_labels=100, dataset='mnist_2layer', n_z=50, n_hidden=(300,), seed=seed, alpha=0.1,
    #                    n_minibatches=100, comment='')
    # learn_yz_x_ss.main(3000, n_labels>100, dataset='mnist_2layer', n_z=50, n_hidden=(500,), seed=seed, alpha=0.1,
    #                    n_minibatches=200, comment='')

    FLAGS = {
        'num_iterations': 40000,
        'num_batches': 100,
        'seed': 31415,
        'n_labeled': 100,
        'alpha': 0.1,
        'latent_dim': 50,
        'require_improvement': 10000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10
    }

    genclass = PreTrainedGenerativeClassifier(num_batches=FLAGS['num_batches'], learning_rate=FLAGS['learning_rate'],
                                              beta1=FLAGS['beta1'], beta2=FLAGS['beta2'], alpha=FLAGS['alpha'],
                                              require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                                              n_labeled=FLAGS['n_labeled'],
                                              num_iterations=FLAGS['num_iterations'],
                                              input_dim=FLAGS['input_dim'],
                                              latent_dim=FLAGS['latent_dim'])  # Should be consistent with model being

    with genclass.session:
        genclass.train_test()
