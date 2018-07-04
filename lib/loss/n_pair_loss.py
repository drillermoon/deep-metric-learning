import tensorflow as tf


def n_pair_loss(anchor, positive, l2_reg=0.5):
    """Multi-class N-pair Loss Function
    Inputs:
    - anchor: Tensor, shape [N, output_dim], The feature vector of an anchor. All anchors are distinct.
    - positive: Tensor, shape [N, output_dim], The feature vector of an anchor. All anchors are distinct.
    - l2_reg: Scalar, A regularization coefficient.

    Return:
    - loss: Tensor, shape [1], The N-pair loss.
    """
    with tf.name_scope('n_pair_loss'):
        n = anchor.shape[0]
        logits = tf.matmul(anchor, tf.transpose(positive))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.eye(n), logits=logits, dim=1))
        anchor_norm = tf.norm(anchor, axis=1)
        positive_norm = tf.norm(positive, axis=1)
        total_norm = tf.concat([anchor_norm, positive_norm], axis=0)
        l2_loss = tf.reduce_mean(total_norm)
        loss += l2_reg*l2_loss
        return loss
