import tensorflow as tf
import sys
sys.path.append('../src')
sys.path.append('..')
slim = tf.contrib.slim
from src import deepMonkeyData
from src import vgg16
import matplotlib.pyplot as plt
import time

model_file = '../Models/-1500'
num_classes = 10
input_file = '/mnt/hgfs/VM_share/deepMonkey224_9606_test.hdf5'
dataset = deepMonkeyData.DataSet(input_file, num_classes)

x_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='X_inputs')
y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
y_pred, _ = vgg16.vgg_16(x_inputs, num_classes=num_classes, is_training=False, scope='vgg_16')
y_label = tf.cast(tf.argmax(y_pred, 1), tf.int32) 
restorer = tf.train.Saver()
init = tf.global_variables_initializer()
configproto = tf.ConfigProto()
configproto.gpu_options.allow_growth = True
configproto.allow_soft_placement = True
with tf.Session(config=configproto) as sess:
    sess.run(init)
    restorer.restore(sess, model_file)
    print('model loaded')

    for i in range(30):
        batch_xs, batch_ys = dataset.next_batch(1)
        print(batch_ys)
        #plt.imshow(batch_xs.reshape((224, 224, 3)))
        #plt.show()
        y = sess.run(y_label, feed_dict={x_inputs: batch_xs})
        print('prediction: %s' % y)

