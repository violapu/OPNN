import tensorflow as tf


def l2_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true))) / tf.sqrt(tf.reduce_sum(tf.square(y_true)))


def max_error(y_true, y_pred):
    return tf.reduce_max(y_pred - y_true) / tf.reduce_max(y_true)


dependencies = {
    'l2_error': l2_error,
    'max_error': max_error
}
