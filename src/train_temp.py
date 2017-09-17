import tensorflow as tf
import sys
sys.path.append('../src')
sys.path.append('..')
slim = tf.contrib.slim
from src import deepMonkeyData
from src import vgg16
import matplotlib.pyplot as plt
import time

tf.device('/gpu:1')
# specify where the pre_trained model is
pre_trained_model_path = '../Models/VGG_pretrain/vgg_16.ckpt'
# specify where the new model will be restored
new_model_path = '../Models/'

num_classes = 10
training_iters = 5000
batch_size = 64 
learning_rate = 0.001
lr_decay = 0.9
data_file_name = "../Data/deepMonkey224.hdf5"
dataset = deepMonkeyData.DataSet(data_file_name, num_classes)

# print variables in the pre_trained model
"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name=pre_trained_model_path, tensor_name='', all_tensors='')
"""

x_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='X_inputs')
y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
lr = tf.placeholder(tf.float32, name='lr')
y_pred, _ = vgg16.vgg_16(x_inputs,
                      num_classes=num_classes,
                      is_training=True,
                      dropout_keep_prob=0.5,
                      scope='vgg_16')
variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
print(variables_to_restore)
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32),
                              tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]),
                                                                     logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)



restorer = tf.train.Saver(variables_to_restore)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
configproto = tf.ConfigProto()
configproto.gpu_options.allow_growth = True
configproto.allow_soft_placement = True
with tf.Session(config=configproto) as sess:
    sess.run(init)
    restorer.restore(sess, pre_trained_model_path)
    print("model restored")

    step = 1
    _lr = learning_rate
    while step < training_iters:
        if step % 200 == 0:
            _lr = _lr * lr_decay
        print('EPOCH %d, lr=%g' % (step, _lr))
        start_time = time.time()
        batch_xs, batch_ys = dataset.next_batch(batch_size, sparse=False)
        sess.run(optimizer, feed_dict={x_inputs: batch_xs, y_inputs: batch_ys, lr: _lr})
        if step % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x_inputs: batch_xs, y_inputs: batch_ys, lr: _lr
            })
            print("step %d, training accuracy %g, speed: %g s/step"
                  %(step, train_accuracy, time.time()-start_time))
        if step % 100 == 0:
            save_path = saver.save(sess, new_model_path, global_step=step)
            print('the save path is ', save_path)
        step += 1


"""
# show image
data_file_name = "/home/leo/Data/DeepMonkey/deepMonkey224.hdf5"

dataset = deepMonkeyData.DataSet(data_file_name, num_classes)
for i in range(20):
    batch_xs, batch_ys = dataset.next_batch(1)
    plt.imshow(batch_xs.reshape((224, 224, 3)) / 255)
    plt.show()
"""
