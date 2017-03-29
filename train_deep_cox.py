from models.deep_cox_propotional.deep_cox import DeepCox

if __name__ == '__main__':
    FLAGS = {
        'num_iterations': 3,  # should 3000 epochs (30000/100)
        'batch_size': 200,
        'seed': 31415,
        'require_improvement': 5000,  # 50 epochs
        'n_train': 50000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'num_classes': 10
    }
    deep_cox = DeepCox(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'], beta1=FLAGS['beta1'],
                       beta2=FLAGS['beta2'], require_improvement=FLAGS['require_improvement'],
                       num_iterations=FLAGS['num_iterations'], seed=FLAGS['seed'])
    with deep_cox.session:
        deep_cox.train_test()
