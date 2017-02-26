import tensorflow as tf
from sklearn import cross_validation
from sklearn.svm import SVC


def train_svm_classifier(features, labels):
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)
    # request probability estimation
    svm = SVC(probability=True)

    svm.fit(X_train, y_train)

    hard_pred = svm.predict(X_test)
    correctly_class = sum(hard_pred == y_test)
    accuracy = correctly_class / len(hard_pred)
    return 1 - accuracy, svm


def softmax_classifier(logits, y_true):
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    return cross_entropy, y_pred_cls
