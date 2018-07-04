from math import pi
from lib.loss.triplet_loss import *
from lib.loss.n_pair_loss import *


def angular_loss_with_triplet(anchor, positive, negative, margin, l2_reg=0.5, alpha=pi/2, weight=2):
    loss = triplet_loss(anchor, positive, negative, margin, l2_reg)
    center = (anchor+positive)/2
    d_ap = tf.reduce_sum(tf.square(anchor-positive), axis=1)
    d_nc = tf.reduce_sum(tf.square(negative-center), axis=1)
    angular_loss = tf.reduce_mean(tf.maximum(0., d_ap-4.0*tf.square(tf.tan(alpha))*d_nc))
    loss += weight*angular_loss
    return loss


def angular_loss_with_n_pair(anchor, positive, l2_reg=0.5, alpha=pi/2, weight=2):
    n = anchor.shape[0]
    loss = n_pair_loss(anchor, positive, l2_reg=l2_reg, alpha=alpha)
    logits = tf.matmul(anchor+positive, tf.transpose(positive))
    logits -= tf.diag(tf.reduce_sum(tf.multiply(anchor+positive, positive)), axis=1)
    logits *= 4*tf.square(tf.tan(alpha))
    logits += 2*(1+tf.square(tf.tan(alpha)))*tf.diag(tf.reduce_sum(tf.multiply(anchor, positive), axis=1))
    angular_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.eye(n), logits=logits, dim=1))
    loss += weight*angular_loss
    return loss
