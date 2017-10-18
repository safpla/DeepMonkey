import tensorflow as tf
import sys
sys.path.append('../src')
sys.path.append('..')
slim = tf.contrib.slim
from src import deepMonkeyData
from src import vgg16
import matplotlib.pyplot as plt
import time

class Predictor():
    def __init__(self):
        num_classes = 10
        x_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='X_inputs')
        is_training_placeholder = tf.placeholder(tf.bool)
        y_pred, _ = vgg16.vgg_16(x_inputs,
                num_classes=num_classes,
                is_training=is_training_placeholder,
                dropout_keep_prob=0.5,
                scope='vgg_16')
        y_label = tf.cast(tf.argmax(y_pred, 1), tf.int32)
        self.x_inputs = x_inputs
        self.is_training_placeholder = is_training_placeholder
        self.y_label = y_label

    def load_model(self, sess, model_file):
        restorer = tf.train.Saver()
        restorer.restore(sess, model_file)
        print('model loaded')

    def predict(self, sess, batch_xs):
        feed_dict = {self.x_inputs: batch_xs, self.is_training_placeholder: False}
        label = sess.run(self.y_label, feed_dict=feed_dict)
        return label

