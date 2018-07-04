import tensorflow as tf


def triplet_loss(anchor, positive, negative, margin, l2_reg=0.5):
    """Triplet Loss Function
    Inputs:
    - anchor: Tensor, shape [batch_size, output_dim], The feature vector of an anchor.
    - positive: Tensor, shape [batch_size, output_dim], The feature vector of an positive sample.
    - negavie: Tensor, shape [batch_size, output_dim], The feature vector of an negative sample.
    - margin: Scalar, A hinge loss margin.
    - l2_reg: Scala, A regularization coefficient.

    Returns:
    - loss: Tensor, shape [1], The triplet loss.
    """
    with tf.name_scope('triplet_loss'):
        d_ap = tf.reduce_sum(tf.square(anchor-positive), axis=1)
        d_an = tf.reduce_sum(tf.square(anchor-negative), axis=1)
        loss = tf.reduce_mean(tf.maximum(0., d_ap-d_an+margin))
        anchor_norm = tf.norm(anchor, axis=1)
        positive_norm = tf.norm(positive, axis=1)
        negative_norm = tf.norm(negative, axis=1)
        total_norm = tf.concat([anchor_norm, positive_norm, negative_norm], axis=0)
        l2_loss = tf.reduce_mean(total_norm)
        loss += l2_reg*l2_loss
        return loss

