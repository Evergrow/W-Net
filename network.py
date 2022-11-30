import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None

def attention(x, x_a, channels, mask, name='attention'):
    with tf.variable_scope(name):
        xs = x.get_shape().as_list()
        mask = tf.image.resize_images(mask, [xs[1], xs[2]])
        query = tf.reshape(x_a, [xs[0], xs[1] * xs[2], xs[3]])
        key = tf.transpose(query, [0, 2, 1])
        x = (1 - mask) * x + mask * x_a
        value = tf.reshape(x, [xs[0], xs[1] * xs[2], xs[3]])
        weight = tf.matmul(query, key)
        weight = tf.nn.softmax(weight, axis=-1)
        o = tf.reshape(tf.matmul(weight, value), xs)
        o = conv(o, channels, kernel=1, stride=1, rate=1, pad=0, use_bias=False)
        x = x + mask * o

        return x, weight


def se_newblock(x1, x2, name='se_layer', reuse=False):
    with tf.variable_scope(name, reuse):
        w = tf.reduce_mean(x1, axis=[1, 2], keepdims=True)
        xs = x2.get_shape().as_list()
        w = linear(w, int(xs[-1] // 2), use_bias=True, name='linear1', reuse=reuse)
        w = relu(w)
        w = linear(w, xs[-1], use_bias=True, name='linear2', reuse=reuse)
        w = sigmoid(w)
        x = w * x2

        return x


def conv(x, channels, kernel=4, stride=2, rate=1, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, dilation_rate=rate, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                       strides=stride, padding='SAME', use_bias=use_bias)

        return x


def linear(x, channels, use_bias, name='linear', reuse=False):
    with tf.variable_scope(name, reuse):
        x = tf.layers.dense(x, units=channels, activation=None, use_bias=use_bias, reuse=reuse,
                            kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)

    return x


def flatten(x):
    return tf.layers.flatten(x)


def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def batch_norm(x, scope='batch_norm', is_train=True):
    return tf_contrib.layers.batch_norm(x,
                                        epsilon=1e-05,
                                        center=True, scale=True,
                                        is_training=is_train,
                                        scope=scope)

