# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/18 23:01 '

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#定义训练参数
learning_rate = 0.01
training_epochs = 20#最大训练回合数，n_samples/batch_size为1个回合
batch_size = 256#每一批次的样本数量
display_step = 1#输出训练结果的间隔
example_to_show = 10

#网络模型参数
n_hidden1_units = 256#第一隐藏层的神经元个数，让编码器和解码器都有相同规模的隐藏层，
n_hidden2_units = 128#第二隐藏层的神经元个数，让编码器和解码器都有相同规模的隐藏层，
n_input_units = 784
n_output_units = n_input_units#解码器输出层神经元个数必须等于输入层数据的units数量

#根据输入输出节点的数量返回初始化好的指定好名称的权重
def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out], dtype=tf.float32, name=name_str))

#根据输出节点的数量返回初始化好的指定好名称的偏置
def BiasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out], dtype=tf.float32, name=name_str))

#构建编码器
def Encoder(x_origin, activate_func=tf.nn.sigmoid):
    #编码器第一隐藏层
    with tf.name_scope('layer1'):
        weights = WeightsVariable(n_input_units, n_hidden1_units, 'weights_encode1')
        biases = BiasesVariable(n_hidden1_units, 'biases_encode1')
        x_code1 = activate_func(tf.add(tf.matmul(x_origin, weights),biases))#y=wx+b得到了256维的编码----编码后的特征码1
    # 编码器第二隐藏层
    with tf.name_scope('layer2'):
        weights = WeightsVariable(n_hidden1_units, n_hidden2_units, 'weights_encode2')
        biases = BiasesVariable(n_hidden2_units, 'biases_encode2')
        x_code2 = activate_func(tf.add(tf.matmul(x_code1, weights), biases))  # y=wx+b得到了128维的编码----编码后的特征码2
    return x_code2#返回第二隐藏层后的特征码



#构建解码器
def Decoder(x_code, activate_func=tf.nn.sigmoid):
    # 解码器第一隐藏层
    with tf.name_scope('layer1'):
        weights = WeightsVariable(n_hidden2_units, n_hidden1_units, 'weights_decode1')#128维---256维
        biases = BiasesVariable(n_hidden1_units, "biases_decode1")#256
        x_decode1 = activate_func(tf.add(tf.matmul(x_code, weights), biases))  # y=wx+b得到了256维的解码---解码后的特征码
    # 解码器第二隐藏层
    with tf.name_scope('layer2'):
        weights = WeightsVariable(n_hidden1_units, n_output_units, 'weights_decode1')  # 256维---784维
        biases = BiasesVariable(n_output_units, "biases_decode2")
        x_decode2 = activate_func(tf.add(tf.matmul(x_decode1, weights), biases))  # y=wx+b得到了784维解码---解码后的特征码
    return x_decode2 #使得x_decode特征码数据与x_original源数据 误差尽可能的小

#调用上面的函数构造计算图
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('X_Origin'):
        X_Origin = tf.placeholder(tf.float32, [None, n_input_units])
    #构建遍码器模型
    with tf.name_scope('Encoder'):
        X_code = Encoder(X_Origin, tf.nn.sigmoid)#可以改为softplus
    #构建解码器模型
    with tf.name_scope('Decoder'):
        X_decode = Decoder(X_code, activate_func=tf.nn.sigmoid)#可以改为softplus
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~Forward prediction finished~~~~~~~~~~~~~~~~~~~~~~~~~")

    #定义Loss损失节点：重构数据与原始数据的误差平方和损失
    with tf.name_scope('Loss'):
        Loss = tf.reduce_mean(tf.pow(X_Origin - X_decode, 2))
    #定义优化器，训练节点
    with tf.name_scope('Train'):#反向求导计算梯度，应用优化器更新每个节点的weight和biase
        Optimizer = tf.train.RMSPropOptimizer(learning_rate)
        Train = Optimizer.minimize(Loss)
    #为所有的变量添加初始化节点
    Init = tf.global_variables_initializer()

    print("~~~~~~~~~~~~~~~~write into calculation graph， show on tensorflow~~~~~~~~~~~~~~~~")#虽然没有启动会话，但是还是可以生成计算图
    summary_write = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    summary_write.flush()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~运行计算图~~~~~~~~~~~~~~~~~~~~~~~~")
    # 加载数据集
    mnist = input_data.read_data_sets('../Basic_Model/mnist_data/', one_hot=True)
    #产生会话session，启动计算图
    with tf.Session() as sess:
        sess.run(Init)
        total_batch = int(mnist.train.num_examples/batch_size)#返回总共需要多少样本批次
        #训练指定回合数，每一回合包含total_batch个批次
        for epoch in range(training_epochs):
            #每一个回合都要把所有的batch跑一边
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                #运行优化器Train节点（backprop）和 Loss节点（获取损失值）
                _, loss = sess.run([Train, Loss], feed_dict={X_Origin:batch_xs})
            #每一轮训练完成后，输出logs
            if epoch % display_step == 0:
                print("Epoch:", '%04d' %(epoch+1), "loss:", '{:.9f}'.format(loss))
        #关闭summary_writer
        summary_write.close()
        print("~~~~~~~~~~~~~~~~~~~~~~training model completely~~~~~~~~~~~~~~~~~~~~~~~~")

        #把训练好的编码器-解码器模型用在测试集上，输出重构后的样本数据
        reconstruction = sess.run(X_decode, feed_dict={X_Origin: mnist.test.images[:example_to_show]})
        #比较原始图像与重构后的图像
        f, a = plt.subplots(2,10, figsize=(10,2))#两行，每行10张图像
        for i in range(example_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))#将1*784维的图像转换成28*28
            a[1][i].imshow(np.reshape(reconstruction[i], (28,28)))
        f.show()
        plt.show()
        plt.waitforbuttonpress()

