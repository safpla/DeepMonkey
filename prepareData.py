import h5py
import tensorflow as tf


filename = "deepMonkey.hdf5"
f = h5py.File(filename,"r")
dataset = f["monkeyData"]
labelset = f["monkeyLabel"]

learningRate = 0.001
trainingIters = 1000
batchSize = 100
displayStep = 20
height = 256
width = 200
numChannel = 3
numClasses = 10
dropout = 0.6

# define placeholder
x = tf.placeholder(tf.type.float32, [batchSize, height, width, numChannel])
y = tf.placeholder(tf.type.float32, [batchSize, numClasses])
keep_prob = tf.placeholder(tf.type.float32)


# convolution
def conv2d(name, X, W, b, strides):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W, strides=strides, padding='SAME'), b), name=name)


# max_pooling
def max_pool(name, X, kernel, strides):
    return tf.nn.max_pool(X, ksize=kernel, strides=strides, padding='SAME', name=name)


# normalization
def norm(name, X, lsize=5):
    return tf.nn.lrn(X, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


# define the net
def alex_net(_X, _weights, _biases, _dropout):
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], strides=[1, 1, 1, 1])
    norm1 = norm('norm1', conv1, lsize=5)

    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'], strides=[1, 1, 1, 1])
    pool2 = max_pool('pool2', conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    norm2 = norm('norm2', pool2, lsize=5)

    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'], strides=[1, 4, 4, 1])

    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], strides=[1, 2, 1, 1])
    pool4 = max_pool('pool4', conv4, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv5 = conv2d('conv5', pool4, _weights['wc5'], _biases['bc5'], strides=[1, 1, 1, 1])
    pool5 = max_pool('pool5', conv5, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    flatten = tf.reshape(pool5, [-1, _weights['fc6'].get_shape().as_list()[0]])

    drop6 = tf.nn.dropout(flatten, dropout)

    fc6 = tf.nn.bias_add(tf.matmul(drop6, _weights['fc6']), _biases['fc6'])
    fc_relu6 = tf.nn.relu(fc6, name='fc6')

    fc7 = tf.nn.bias_add(tf.matmul(fc_relu6, _weights['fc7']), _biases['fc7'])
    fc_relu7 = tf.nn.relu(fc7, name='fc7')

    fc8 = tf.nn.bias_add(tf.matmul(fc_relu7, _weigths['fc8']), _biases['fc8'], name='fc8')

    return fc8


# define net parameters
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([4, 4, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([4, 4, 128, 256])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'fc6': tf.Variable(tf.random_normal([3*3*256, 1024])),
    'fc7': tf.Variable(tf.random_normal([1024, 512])),
    'fc8': tf.Variable(tf.random_normal([512, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'fc6': tf.Variable(tf.random_normal([1024])),
    'fc7': tf.Variable(tf.random_normal([512])),
    'fc8': tf.Variable(tf.random_normal([10]))
}


# build the model
pred = alex_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < trainingIters:
        batch_xs, batch_ys = m
