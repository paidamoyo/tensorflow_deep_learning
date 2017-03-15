import tensorflow as tf


def softmax_classifier(logits, y_true):
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    return cross_entropy, y_pred_cls


def svm_classifier(weights, logits, svmC, y_true):
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(weights))
    print(logits)
    hinge = tf.reduce_sum(tf.maximum(tf.zeros([100, 10]),
                                     tf.ones([100, 10]) - tf.multiply(y_true, logits)))
    svm_loss = regularization_loss + svmC * hinge
    return svm_loss, y_pred_cls
