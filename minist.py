
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
                 use_bais=False,
                 keep_step=10):
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
        self.local_step = 0
        self.kernel_norm = None
        self.keep_step = keep_step

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
        input_norm = k.sqrt(tf.nn.conv2d(inputs * inputs, ones_filter, self.strides, padding=self.padding) + eps)
        return input_norm

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
        self.radius = self.add_variable(name='radius',
                                        shape=[1, 1, 1, self.filtes],
                                        initializer=tf.keras.initializers.Constant(1.))

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
        inputs *= tf.log1p(inputs_norm)
        # inputs *= k.minimum(inputs_norm, self.radius)/self.radius
        # inputs *= tf.tanh(kernel_norm / (self.radius**2 + eps)) * tf.tanh(inputs_norm / (self.radius**2 + eps))
        if self.use_bais:
            inputs = tf.nn.bias_add(inputs, self.bais)
        self.local_step += 1
        if self.local_step == self.keep_step:
            tf.assign(self.kernel, self.kernel / kernel_norm)
            self.local_step = 0

        return inputs


class DC3(tf.keras.Model):
    def __init__(self):
        super(DC3, self).__init__()
        self.conv1 = BaseDcLayer(filters=32, kernel_size=3, strides=2)
        self.conv2 = BaseDcLayer(filters=64, kernel_size=3, strides=1)
        self.conv3 = BaseDcLayer(filters=128, kernel_size=3, strides=2)
        self.conv4 = BaseDcLayer(filters=256, kernel_size=3, strides=2)
        self.conv5 = BaseDcLayer(filters=512, kernel_size=3, strides=1)
        self.conv6 = BaseDcLayer(filters=40, kernel_size=3, strides=2, padding="valid")
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=True, mask=None):
        vector = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(inputs, training), training), training), training), training), training)
        vector_ = tf.squeeze(vector)
        vector = self.dense(vector_)
        return vector, vector_


class Convbnrelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding="same", activation="relu"):
        super(Convbnrelu, self).__init__()
        if activation == "relu":
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = tf.keras.layers.LeakyReLU(alpha=1)
        self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filters,
                                                                 kernel_size=kernel_size,
                                                                 strides=strides,
                                                                 padding=padding,
                                                                 use_bias=False),

                                          self.activation])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training)


class DC32(tf.keras.Model):
    def __init__(self):
        super(DC32, self).__init__()
        self.conv1 = Convbnrelu(filters=32, kernel_size=3, strides=2)
        self.conv2 = Convbnrelu(filters=64, kernel_size=3, strides=1)
        self.conv3 = Convbnrelu(filters=128, kernel_size=3, strides=2)
        self.conv4 = Convbnrelu(filters=256, kernel_size=3, strides=2)
        self.conv5 = Convbnrelu(filters=512, kernel_size=3, strides=1)
        self.conv6 = Convbnrelu(filters=40, kernel_size=3, strides=2, padding="valid", activation="sigmoid")
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=True, mask=None):
        vector = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(inputs, training), training), training), training), training), training)
        vector_ = tf.squeeze(vector)
        vector = self.dense(vector_)
        return vector, vector_


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition.pca import PCA
    pca = PCA(n_components=2, copy=True, whiten=True)
    tf.enable_eager_execution()
    optimizer = tf.train.AdamOptimizer(1e-2)
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    train_x = mnist.train.images / 255.
    train_y = mnist.train.labels
    test_x = mnist.test.images / 255.
    test_y = mnist.test.labels
    train_x = train_x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000) \
        .batch(1024, drop_remainder=True).repeat(5)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(1024)
    model = DC32()  # 32 97.8 3 96
    train_step = 0
    for train_step, data in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits, repsentation = model(data[0])
            la = tf.cast(data[1], tf.float32)
            # loss = k.mean(k.abs(tf.matmul(repsentation, repsentation, transpose_b=True) - tf.matmul(la, la,
                                                                                                    # transpose_b=True))**2)
            loss = k.mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=la))
            grd = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grd, model.trainable_variables))
            print(loss)

    for select_data, select_la in train_dataset:
        logits, repsentation = model(select_data, False)
        repsentation = pca.fit_transform(repsentation.numpy())
        # repsentation = repsentation.numpy()
        logits = np.argmax(logits, axis=-1)
        for i in range(10):
            pts = repsentation[np.argmax(select_la, axis=-1) == i]
            plt.scatter(x=pts[:, 0], y=pts[:, 1])
        plt.show()
        break
