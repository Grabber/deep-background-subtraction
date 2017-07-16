from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2

# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
# tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
# tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
# tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
# tf.flags.DEFINE_float('learning_rate', '0.01', '')
# tf.flags.DEFINE_string('mode', 'train', 'Mode train, val')

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

def network(image, num_classes=2, dropout_prob=0.5, is_training=False):
    with tf.variable_scope('network'):
        conv_1 = tf.layers.conv2d.conv2d(image, 32, [5, 5], scope='conv_1')
        pool_1 = tf.layers.conv2d.max_pool2d(conv_1, [2, 2], 2, scope='pool_1')
        conv_2 = tf.layers.conv2d.conv2d(pool_1, 64, [5, 5], scope='conv_2')
        pool_2 = tf.layers.conv2d.max_pool2d(conv_2, [2, 2], 2, scope='pool_2')
        fc_1 = tf.layers.conv2d.fully_connected(pool_2, 1024, scope='fc_1')
        dropout_1 = tf.layers.conv2d.dropout(fc_1, dropout_prob, is_training=is_training, scope='dropout_1')
        fc_2 = tf.layers.conv2d.fully_connected(dropout_1, num_classes, activation_fn=None, scope='fc_2')

    return fc_2

# def main(argv=None):
#     learning_rate = tf.placeholder(tf.float32, name='learning_rate')
#     images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
#     labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
#     is_train = tf.placeholder(tf.bool, name='is_train')
#     global_step = tf.Variable(0, name='global_step', trainable=False)
#     weight_decay = 0.0005

#     # with slim.arg_scope(
#     #     [slim.conv2d, slim.fully_connected],
#     #     weights_regularizer=slim.l2_regularizer(weight_decay),
#     #     weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
#     #     activation_fn=tf.nn.relu) as sc:
#     #   return sc