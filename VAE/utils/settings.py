def initialize():
    FLAGS = {
        'data_directory': 'data/MNIST/',
        'summaries_dir': 'summaries/',
        'save_path': 'results/train_weights',
        'test_batch_size': 200,
        'num_iterations': 40000,
        'num_batches': 100,
        'seed': 12000,
        'n_labeled': 100,
        'alpha': 0.1,
        'm1_h_dim': 500,
        'm2_h_dim': 500,
        'latent_dim': 50,
        'require_improvement': 30000,
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'input_dim': 28 * 28,
        'num_classes': 10
    }
    return FLAGS
