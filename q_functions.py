import tensorflow as tf
from tensorflow.python.layers.layers import *


class QFunction:
    def __init__(self, state, n_actions, scope):
        raise NotImplementedError

    def is_recurrent(self):
        raise NotImplementedError


class CartPoleNet(QFunction):
    def __init__(self, state, n_actions, scope):
        hidden = flatten(state) # flatten to make sure 2-D

        with tf.variable_scope(scope):
            hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
            hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
            qvalues = dense(hidden, units=n_actions, activation=None)

        self.qvalues = qvalues

    def is_recurrent(self):
        return False


class AtariRecurrentConvNet(QFunction):
    def __init__(self, state, n_actions, scope):
        state = tf.cast(state, tf.float32) / 255.0

        hidden = tf.reshape(state, [-1, state.shape[2], state.shape[3], state.shape[4]])
        print('Recurrent', state.shape)

        with tf.variable_scope(scope):
            hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            hidden = tf.reshape(hidden, [tf.shape(state)[0], state.shape[1], tf.size(hidden[0])])

            cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
            self.rnn_state = cell.zero_state(tf.shape(state)[0], tf.float32)
            hidden, new_rnn_state = tf.nn.dynamic_rnn(cell, inputs=hidden, initial_state=self.rnn_state, dtype=tf.float32)

            qvalues = dense(hidden[:, -1], units=n_actions, activation=None)

        self.qvalues = qvalues

    def is_recurrent(self):
        return True


class AtariRecurrentConvNet2(QFunction):
    def __init__(self, state, n_actions, scope):
        state = tf.cast(state, tf.float32) / 255.0

        hidden = tf.reshape(state, [-1, state.shape[2], state.shape[3], state.shape[4]])
        print('Recurrent', state.shape)

        with tf.variable_scope(scope):
            hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            hidden = tf.reshape(hidden, [tf.shape(state)[0], state.shape[1], tf.size(hidden[0])])

            from lru_cell import LRUCell
            cell = LRUCell(num_units=1582)
            #from lru_cell import AddressableMemoryArray
            #cell = AddressableMemoryArray(num_units=512)
            self.rnn_state = cell.zero_state(tf.shape(state)[0], tf.float32)
            hidden, new_rnn_state = tf.nn.dynamic_rnn(cell, inputs=hidden, initial_state=self.rnn_state, dtype=tf.float32)

            qvalues = dense(hidden[:, -1], units=n_actions, activation=None)

        self.qvalues = qvalues

    def is_recurrent(self):
        return True


class AtariConvNet(QFunction):
    def __init__(self, state, n_actions, scope):
        state = tf.cast(state, tf.float32) / 255.0

        hidden = state
        hidden = tf.unstack(hidden, axis=1)
        hidden = tf.concat(hidden, axis=-1)
        print('Feedforward', hidden.shape)

        with tf.variable_scope(scope):
            hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            hidden = flatten(hidden)

            hidden  = dense(hidden, units=512,       activation=tf.nn.relu)
            qvalues = dense(hidden, units=n_actions, activation=None)

        self.qvalues = qvalues

    def is_recurrent(self):
        return False
