
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras import backend as k
import tensorflow as tf

eps = 1e-12


def orthogonal_regularizer(scale=1e-5):
    """ Defining the Orthogonal regularizer and return the function at
     last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w):
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        shape = w.get_shape().as_list()
        c = shape[-1]

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_mul = tf.matmul(w, w, transpose_a=True)
        reg = w_mul - identity

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


class BaseDcLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding="same",
                 initializer=tf.keras.initializers.random_normal(0, .02),
                 regularizer=orthogonal_regularizer(),
                 use_bais=True):
        super(BaseDcLayer, self).__init__()
        self.filtes = filters
        self.kernel_size = kernel_size
        self.strides = (1, strides, strides, 1)
        self.padding = padding.upper()
        self.initializer = initializer
        self.regularizer = regularizer
        self.kernel = None
        self.bais = None
        self.use_bais = use_bais
        self.chs = None
        self.radius = None
        self.moving_avg_norm = None
        self.local_step = -1

    def _get_filter_norm(self):

        return k.sqrt(k.sum(self.kernel * self.kernel, axis=[0, 1, 2], keepdims=True) + eps)

    def _get_input_norm(self, inputs):
        """
        The norm of input X should be each conv response area, so that |X|
        is implement by conv(ones, inputs**2)
        :param self:
        :param inputs:
        :return: norm for each response area [bs, h - (k-1)+2p//strides, ..., 1]
        """
        ones_filter = tf.ones([self.kernel_size, self.kernel_size, self.chs, 1])
        input_norm = k.sqrt(tf.nn.conv2d(inputs ** 2, ones_filter, self.strides, padding=self.padding) + eps)
        return input_norm

    def x_norm_batch_norm(self, x, phase_training):
        """
        None solution method
        :param x:
        :param phase_training:
        :return:
        """
        batch_mean = k.mean(x, axis=[0, 1, 2])
        if self.moving_avg_norm is None:
            self.moving_avg_norm = tf.Variable(batch_mean, trainable=False, name="moving_avg")
        elif phase_training:
            self.moving_avg_norm = batch_mean * .001 + self.moving_avg_norm * .999
        return self.moving_avg_norm

    def build(self, input_shape):
        self.chs = input_shape[-1]
        self.kernel = self.add_variable(shape=[self.kernel_size, self.kernel_size, self.chs, self.filtes],
                                        initializer=self.initializer,
                                        regularizer=self.regularizer,
                                        name="kernel")
        if self.use_bais:
            self.bais = self.add_variable(name='bias',
                                          shape=[self.filtes],
                                          initializer=tf.keras.initializers.Constant(0.))
        self.radius = self.add_variable(name="radius", shape=None,
                                        initializer=tf.keras.initializers.Constant(1))

    @staticmethod
    def normalize(kernel, norm):
        return kernel / norm

    def call(self, inputs, training=None, mask=None):
        inputs_norm = self._get_input_norm(inputs)
        kernel_norm = self._get_filter_norm()
        inputs = tf.nn.conv2d(inputs,
                              self.normalize(self.kernel, kernel_norm),
                              self.strides,
                              self.padding,
                              data_format="NHWC")
        inputs /= inputs_norm
        inputs = 1 - tf.acos(inputs) * 2 / np.pi
        # inputs = tf.sign(inputs) * inputs ** 2
        # inputs *= tf.log1p(1 + inputs_norm)
        # inputs *= (1 - tf.nn.relu(1 - inputs_norm))/inputs_norm
        inputs *= tf.tanh(kernel_norm / self.radius) * tf.tanh(inputs_norm / self.radius)
        if self.use_bais:
            inputs = tf.nn.bias_add(inputs, self.bais)
        # self.kernel = self.kernel / kernel_norm
        return inputs


class DC3(tf.keras.Model):
    def __init__(self):
        super(DC3, self).__init__()
        self.model = tf.keras.Sequential([BaseDcLayer(filters=32, kernel_size=3, strides=2),
                                          BaseDcLayer(filters=64, kernel_size=3, strides=2),
                                          BaseDcLayer(filters=128, kernel_size=3, strides=2),
                                          tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(units=10)])

    def call(self, inputs, training=True, mask=None):
        return self.model(inputs, training)


class Convbnrelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(Convbnrelu, self).__init__()
        self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filters,
                                                                 kernel_size=kernel_size,
                                                                 strides=strides,
                                                                 padding="same",
                                                                 use_bias=False),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.ReLU()])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training)


class DC32(tf.keras.Model):
    def __init__(self):
        super(DC32, self).__init__()
        self.model = tf.keras.Sequential([Convbnrelu(filters=32, kernel_size=3, strides=2),
                                          Convbnrelu(filters=64, kernel_size=3, strides=2),
                                          Convbnrelu(filters=128, kernel_size=3, strides=2),
                                          tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(units=10)])

    def call(self, inputs, training=True, mask=None):
        return self.model(inputs, training)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    tf.enable_eager_execution()
    optimizer = tf.train.AdamOptimizer(1e-2)
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    train_x = train_x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000) \
        .batch(256, drop_remainder=True).repeat(20)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(1024)
    model = DC3()  # 32 97.8 3 96
    train_step = 0
    for train_step, data in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(data[0])
            loss = k.mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data[1]))
            grd = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grd, model.trainable_variables))
            # print(loss)
        if train_step % 600 == 0:
            loss = []
            for select_data, select_la in test_dataset:
                logits = model(select_data, False)
                logits = np.argmax(logits, axis=-1)
                select_la = np.argmax(select_la, axis=-1)
                loss.append(np.mean(logits == select_la))
            print(np.mean(loss))
