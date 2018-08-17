import tensorflow as tf
from model import Model, Config
import os

if __name__ == "__main__":
    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 获得数据
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    model = Model(Config)

    # 加载上一次保存的模型
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    model.train(model_path, mnist)
