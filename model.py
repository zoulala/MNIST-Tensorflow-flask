import tensorflow as tf
import numpy as np
import time
import os

class Config(object):
    """CNN配置参数"""
    file_name = 'cnn1'  #保存模型文件

    seq_length = 784  # 序列长度
    num_classes = 10  # 类别数

    num_filters1 = 32  # 卷积核数目
    kernel_size1 = 5  # 卷积核尺寸
    num_filters2 = 64  # 卷积核数目
    kernel_size2 = 5  # 卷积核尺寸


    hidden_dim = 128  # 全连接层神经元

    train_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder('float', shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder('float', shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Cnn模型
        self.cnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def cnn(self):
        """CNN模型"""

        # 权重和偏置初始化函数
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # 卷积和池化
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                                padding='SAME')  # [batch , in_height , in_width, in_channels]，in_height , in_width,移动步长；“SAME”表示采用填充的方式，简单地理解为以0填充边缘，“VALID”表示采用不填充的方式，多余地进行丢弃

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2,
                                           1],
                                  padding='SAME')  # x:[batch, height, width, channels],ksize：表示池化窗口的大小：一个长度为4的一维列表，一般为[1, height, width, 1]，因不想在batch和channels上做池化，则将其值设为1。

        # 第一层卷积，（总参数个数（5*5+1）*通道数*32）
        W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5大小的卷积核，1个通道（灰度图像1通道，RGB为3通道），32个卷积核
        b_conv1 = bias_variable([32])  # 32个偏置

        x_image = tf.reshape(self.input_x, [-1, 28, 28, 1])  # [batch , in_height , in_width, in_channels]，结果大小【28,28,1】

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 总参数个数（5*5+1）*32   。 结果大小【28,28,32】
        h_pool1 = max_pool_2x2(h_conv1)  # 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling.结果大小【14,14,32】

        # 第二层卷积
        W_conv2 = weight_variable([5, 5, 32, 64])  # 第一层32个卷积核产生了32个特征图，因此有32个通道，并用64个卷积核将产生64个特征图
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 结果大小【14,14,64】
        h_pool2 = max_pool_2x2(h_conv2)  # 结果大小【7,7,64】

        # 密集连接层
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 结果大小【1,1024】

        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 输出层
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 结果大小【1,10】

        # 训练和评估模型
        cross_entropy = -tf.reduce_sum(self.input_y * tf.log(self.y_conv))
        self.optim = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # ADAM优化器来做梯度最速下降
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, model_path, mnist):
        with self.session as sess:
            for i in range(20000):
                batch = mnist.train.next_batch(50)
                self.optim.run(feed_dict={self.input_x: batch[0], self.input_y: batch[1], self.keep_prob: 0.5})
                if i % 200 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={self.input_x: batch[0], self.input_y: batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                    self.saver.save(sess, os.path.join(model_path, 'model'))


    def test(self, x ):
        # print("test accuracy %g" % self.accuracy.eval(feed_dict={
        #     self.input_x: mnist.test.images, self.input_y: mnist.test.labels, self.keep_prob: 1.0}))
        y_pre = self.session.run(self.y_conv, feed_dict={self.input_x: x, self.keep_prob:1.0}).flatten().tolist()
        return y_pre


