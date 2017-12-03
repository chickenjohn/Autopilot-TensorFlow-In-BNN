import tensorflow as tf
import scipy


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def binarize(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        x = tf.clip_by_value(x, -1, 1)
        return tf.sign(x)


def HardTanh(x, name='HardTanh'):
    return tf.clip_by_value(x, -1, 1)


class ConvModel(object):
    def __init__(self, drop_out=False, relu=True, is_training=True):
        self.x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

        self.x_image = self.x

        self.keep_prob = tf.placeholder(tf.float32)

        # first convolutional layer
        self.W_conv1 = weight_variable([5, 5, 3, 24])
        self.b_conv1 = bias_variable([24])

        self.h_conv1 = conv2d(self.x_image, binarize(
            self.W_conv1), 2) + self.b_conv1

        self.batch_norm1 = tf.contrib.layers.batch_norm(
            self.h_conv1, is_training=is_training, trainable=True)
        if relu:
            self.batch_norm1 = tf.nn.relu(self.batch_norm1)
        self.out_feature1 = HardTanh(self.batch_norm1)

        # second convolutional layer
        self.W_conv2 = weight_variable([5, 5, 24, 36])
        self.b_conv2 = bias_variable([36])

        self.h_conv2 = conv2d(binarize(self.out_feature1),
                              binarize(self.W_conv2), 2) + self.b_conv2

        self.batch_norm2 = tf.contrib.layers.batch_norm(
            self.h_conv2, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm2 = tf.nn.relu(self.batch_norm2)
        self.out_feature2 = HardTanh(self.batch_norm2)

        # third convolutional layer
        self.W_conv3 = weight_variable([5, 5, 36, 48])
        self.b_conv3 = bias_variable([48])

        self.h_conv3 = conv2d(binarize(self.out_feature2),
                              binarize(self.W_conv3), 2) + self.b_conv3

        self.batch_norm3 = tf.contrib.layers.batch_norm(
            self.h_conv3, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm3 = tf.nn.relu(self.batch_norm3)
        self.out_feature3 = HardTanh(self.batch_norm3)

        # fourth convolutional layer
        self.W_conv4 = weight_variable([3, 3, 48, 64])
        self.b_conv4 = bias_variable([64])

        self.h_conv4 = conv2d(binarize(self.out_feature3),
                              binarize(self.W_conv4), 1) + self.b_conv4

        self.batch_norm4 = tf.contrib.layers.batch_norm(
            self.h_conv4, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm4 = tf.nn.relu(self.batch_norm4)
    
        self.out_feature4 = HardTanh(self.batch_norm4)

        # fifth convolutional layer
        self.W_conv5 = weight_variable([3, 3, 64, 64])
        self.b_conv5 = bias_variable([64])

        self.h_conv5 = conv2d(binarize(self.out_feature4),
                              binarize(self.W_conv5), 1) + self.b_conv5

        self.batch_norm5 = tf.contrib.layers.batch_norm(
            self.h_conv5, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm5 = tf.nn.relu(self.batch_norm5)

        self.out_feature5 = HardTanh(self.batch_norm5)

        # FCL 1
        self.W_fc1 = weight_variable([1152, 1164])
        self.b_fc1 = bias_variable([1164])

        self.out_feature5_flat = tf.reshape(
            binarize(self.out_feature5), [-1, 1152])
        self.h_fc1 = tf.matmul(self.out_feature5_flat,
                               binarize(self.W_fc1)) + self.b_fc1

        self.batch_norm6 = tf.contrib.layers.batch_norm(
            self.h_fc1, is_training=is_training, trainable=True)
        
        if relu:
            self.batch_norm6 = tf.nn.relu(self.batch_norm6)

        self.out_feature6 = HardTanh(self.batch_norm6)
        if drop_out:
            self.out_feature6 = tf.nn.dropout(
                self.out_feature6, self.keep_prob)

        # FCL 2
        self.W_fc2 = weight_variable([1164, 100])
        self.b_fc2 = bias_variable([100])

        self.h_fc2 = tf.matmul(binarize(self.out_feature6),
                               binarize(self.W_fc2)) + self.b_fc2

        self.batch_norm7 = tf.contrib.layers.batch_norm(
            self.h_fc2, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm7 = tf.nn.relu(self.batch_norm7)

        self.out_feature7 = HardTanh(self.batch_norm7)
        if drop_out:
            self.out_feature7 = tf.nn.dropout(
                self.out_feature7, self.keep_prob)

        # FCL 3
        self.W_fc3 = weight_variable([100, 50])
        self.b_fc3 = bias_variable([50])

        self.h_fc3 = tf.matmul(binarize(self.out_feature7),
                               binarize(self.W_fc3)) + self.b_fc3

        self.batch_norm8 = tf.contrib.layers.batch_norm(
            self.h_fc3, is_training=is_training, trainable=True)

        if relu:
            self.batch_norm8 = tf.nn.relu(self.batch_norm8)

        self.out_feature8 = HardTanh(self.batch_norm8)
        if drop_out:
            self.out_feature8 = tf.nn.dropout(
                self.out_feature8, self.keep_prob)

        # FCL 3
        self.W_fc4 = weight_variable([50, 10])
        self.b_fc4 = bias_variable([10])

        self.h_fc4 = tf.matmul(binarize(self.out_feature8),
                               binarize(self.W_fc4)) + self.b_fc4

        self.batch_norm9 = tf.contrib.layers.batch_norm(
            self.h_fc4, is_training=is_training, trainable=True)
        self.out_feature9 = HardTanh(self.batch_norm9)

        if relu:
            self.batch_norm9 = tf.nn.relu(self.batch_norm4)

        if drop_out:
            self.out_feature9 = tf.nn.dropout(
                self.out_feature9, self.keep_prob)

        # Output
        self.W_fc5 = weight_variable([10, 1])
        self.b_fc5 = bias_variable([1])

        # scale the atan output
        self.y = tf.multiply(
            tf.atan(tf.matmul(self.out_feature9, self.W_fc5) + self.b_fc5), 2)
