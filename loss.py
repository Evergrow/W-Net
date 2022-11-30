import tensorflow as tf
import os
import numpy as np
import inspect


def adversarial_loss(outputs, is_real, is_disc=None, type='nsgan'):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    outputs = tf.reshape(outputs, [-1])
    if type == 'hinge':
        if is_disc:
            if is_real:
                outputs = -outputs
            return tf.reduce_mean(tf.nn.relu(1 + outputs))
        else:
            return tf.reduce_mean(-outputs)

    elif type == 'nsgan':
        labels = tf.ones_like(outputs) if is_real else tf.zeros_like(outputs)
        loss = tf.keras.metrics.binary_crossentropy(labels, outputs)
        return loss
    elif type == 'lsgan':
        labels = tf.ones_like(outputs) if is_real else tf.zeros_like(outputs)
        loss = tf.keras.metrics.mean_squared_error(labels, outputs)
        return loss


def seg_loss(pred, mask):
    y_pred = tf.nn.softmax(pred, axis=-1)
    pred1, pred2 = tf.split(y_pred, [1, 1], axis=-1)
    label = tf.to_float(mask)
    log1 = tf.log(tf.clip_by_value(pred1, 1e-8, 1.0 - 1e-8))
    log2 = tf.log(tf.clip_by_value(pred2, 1e-8, 1.0 - 1e-8))
    loss = - tf.transpose(tf.multiply(1 - label, log1), perm=[1, 2, 3, 0]) - \
           tf.transpose(tf.multiply(label, log2), perm=[1, 2, 3, 0])
    loss = tf.transpose(loss, perm=[3, 0, 1, 2])
    loss = tf.reduce_mean(loss)
    return loss

def focal_loss(pred, mask, ratio, gamma=0):
    y_pred = tf.nn.softmax(pred, axis=-1)
    pred1, pred2 = tf.split(y_pred, [1, 1], axis=-1)
    label = tf.to_float(mask)
    log1 = tf.multiply(pred2 ** gamma, tf.log(tf.clip_by_value(pred1, 1e-8, 1.0 - 1e-8)))
    log2 = tf.multiply(pred1 ** gamma, tf.log(tf.clip_by_value(pred2, 1e-8, 1.0 - 1e-8)))
    loss = - ratio * tf.transpose(tf.multiply(1 - label, log1), perm=[1, 2, 3, 0]) - \
           (1 - ratio) * tf.transpose(tf.multiply(label, log2), perm=[1, 2, 3, 0])
    loss = tf.transpose(loss, perm=[3, 0, 1, 2])
    loss = tf.reduce_mean(loss)
    return loss

def l1_loss(inputs, targets):
    inputs = tf.reshape(inputs, [-1])
    targets = tf.reshape(targets, [-1])
    loss = tf.reduce_mean(tf.abs(inputs - targets))
    return loss


def perceptual_loss(x, y, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """
    x_vgg = Vgg16(x)
    y_vgg = Vgg16(y)

    content_loss = 0.0

    content_loss += l1_loss(x_vgg.pool1 / 255, y_vgg.pool1 / 255)
    content_loss += l1_loss(x_vgg.pool2 / 255, y_vgg.pool2 / 255)
    content_loss += l1_loss(x_vgg.pool3 / 255, y_vgg.pool3 / 255)

    return content_loss


VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, rgb, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            # print(path)

        self.data_dict = np.load(vgg16_npy_path, allow_pickle=True, encoding='latin1').item()
        # print("npy file loaded")
        self.build(rgb)

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        # print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
    