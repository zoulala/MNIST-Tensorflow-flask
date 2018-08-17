import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from model import Model, Config


model_path = os.path.join('models', Config.file_name)
model = Model(Config)

# 加载保存的模型
checkpoint_path = tf.train.latest_checkpoint(model_path)
model.load(checkpoint_path)

# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = model.test(input)
    return jsonify(results=[output1,])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)



