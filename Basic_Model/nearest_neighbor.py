# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 15:49 '

import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#导入Mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

#对mnist做数量限制
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)
print("Xtrain.shape: ", Xtrain.shape, ", Xtest.shape: ", Xtest.shape)
print("Ytrain.shape: ", Ytrain.shape, ", Ytest.shape: ", Ytest.shape)

#计算图输入占位符
xtrain = tf.placeholder("float", [None, 784])
xtest = tf.placeholder("float", [784])

# 计算L1距离，进行最近邻计算
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)
# 预测获得最小距离索引，根据最邻近的类标签进行判断
pred = tf.arg_min(distance, 0)
# 评估：给定一条测试样本，看是否预测正确

# 初始化节点
init = tf.global_variables_initializer()

# 最邻近分类器的准确率
accuracy = 0.

# 启动会话
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xtest)  # 测试样本的数量
    # 在测试集上进行循环
    for i in range(Ntest):
        # 获取当前样本的的最近邻
        nn_index = sess.run(pred, feed_dict={xtrain: Xtrain, xtest: Xtest[i, :]})  # 每次只传入一个测试样本
        # 获取最近邻标签，然后与真实标签进行对比
        pred_class_label = np.argmax(Ytrain[nn_index])  # 因为是一个10维的向量，是为了找到那个1
        true_class_label = np.argmax(Ytest[i])
        print("Test", i, "Predicted Class LabelL ", pred_class_label, "True Class Label: ", true_class_label)
        # 计算准确率
        if pred_class_label == true_class_label:
            accuracy += 1
    print("Done!")
    accuracy /= Ntest
    print("Accuracy: ", accuracy)  # 注意计算accuracy使用的numpy实现的，没有运用到Tensorflow的计算图中

