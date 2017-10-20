import tensorflow as tf
import sys
sys.path.append('../src')
sys.path.append('..')
slim = tf.contrib.slim
from src import deepMonkeyData
from src import vgg16
import matplotlib.pyplot as plt
import time

# tf.device('/gpu:1')
# specify where the pre_trained model is
pre_trained_model_path = '../Models/VGG_pretrain/vgg_16.ckpt'
load_previous_model = True
previous_model_path = '../Models/M93A_9606/-700'
# specify where the new model will be restored
new_model_path = '../Models/M93A_9606_tc/'
summaries_dir = '../train_logs'
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)

num_classes = 10
training_iters = 5000
batch_size = 64
learning_rate = 0.0001
lr_decay = 0.9
data_valid_file_name = "../Data/deepMonkey224_M93A_test_new.hdf5"
data_train_file_name = "../Data/deepMonkey224_M93A_train_new.hdf5"
dataset_train = deepMonkeyData.DataSet(data_train_file_name, num_classes)
dataset_valid = deepMonkeyData.DataSet(data_valid_file_name, num_classes)

x_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='X_inputs')
y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
is_training_placeholder = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32, name='lr')
y_pred, _ = vgg16.vgg_16(x_inputs,
                      num_classes=num_classes,
                      is_training=is_training_placeholder,
                      dropout_keep_prob=0.5,
                      scope='vgg_16')
variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
print(variables_to_restore)
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32),
                              tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]),
                                                                    logits=y_pred))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cost', cost)
summary_merged = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)


restorer = tf.train.Saver(variables_to_restore)
restorer1 = tf.train.Saver()
init = tf.global_variables_initializer()

saver = tf.train.Saver()
configproto = tf.ConfigProto()
configproto.gpu_options.allow_growth = True
configproto.allow_soft_placement = True
with tf.Session(config=configproto) as sess:
    sess.run(init)
    restorer.restore(sess, pre_trained_model_path)
    print("pretrained model restored")
    if load_previous_model:
        restorer1.restore(sess, previous_model_path)
    print("previous model restored")

    step = 1
    _lr = learning_rate
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir + '/valid', sess.graph)

    while step < training_iters:
        if step % 200 == 0:
            _lr = _lr * lr_decay
        print('EPOCH %d, lr=%g' % (step, _lr))
        start_time = time.time()
        batch_xs, batch_ys = dataset_train.next_batch(batch_size, sparse=False)
        feed_dict = {x_inputs: batch_xs, y_inputs: batch_ys, is_training_placeholder: True, lr: _lr}
        sess.run(optimizer, feed_dict=feed_dict)
        if step % 10 == 0:
            feed_dict = {x_inputs: batch_xs, y_inputs: batch_ys, is_training_placeholder: False, lr: _lr}
            train_accuracy, train_cost, summary = sess.run([accuracy, cost, summary_merged], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            print("step %d, training accuracy %g, loss %g speed: %g s/step"
                    %(step, train_accuracy, train_cost, time.time()-start_time))
        if step % 100 == 0:
            # validation
            valid_accuracys = 0
            for i in range(int(dataset_valid.num_examples / batch_size)):
                batch_xs_valid, batch_ys_valid = dataset_valid.next_batch(batch_size)
                feed_dict = {x_inputs: batch_xs_valid, y_inputs: batch_ys_valid, is_training_placeholder: False, lr: _lr}
                valid_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                valid_accuracys += valid_accuracy
            valid_accuracy = valid_accuracys / int(dataset_valid.num_examples / batch_size)
            print("step %d, valid accuracy %g"
                  %(step, valid_accuracy))
            save_path = saver.save(sess, new_model_path, global_step=step)
            print('the save path is ', save_path)
        step += 1

"""
# show image
data_train_file_name = "/home/leo/Data/DeepMonkey/deepMonkey224.hdf5"

dataset_train = deepMonkeyData.DataSet(data_train_file_name, num_classes)
for i in range(20):
    batch_xs, batch_ys = dataset_train.next_batch(1)
    plt.imshow(batch_xs.reshape((224, 224, 3)) / 255)
    plt.show()
"""
