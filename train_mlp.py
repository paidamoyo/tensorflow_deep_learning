from models.mlp.mlp_classifier import MLPClassifier

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
        'input_dim': 28 * 28
    }

    mlp = MLPClassifier(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                        beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                        require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                        num_iterations=FLAGS['num_iterations'],
                        num_classes=FLAGS['num_classes'], input_dim=FLAGS['input_dim'])
    with mlp.session:
        mlp.train_test()
