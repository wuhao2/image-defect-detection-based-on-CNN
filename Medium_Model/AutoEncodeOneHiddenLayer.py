# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/19 15:49 '


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
n_hidden_units = 256#隐藏层的神经元个数，让编码器和解码器都有相同规模的隐藏层，可以改为128，编码效率差些，看效果
n_input_units = 784
n_output_units = n_input_units#解码器输出层神经元个数必须等于输入层数据的units数量

#定义一个张量汇总函数（均值，标准差，最大最小值，直方图）
#用于Tensorflow的可视化
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)#统计均值
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))#统计标差
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


#根据输入输出节点的数量返回初始化好的指定好名称的权重
def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out], dtype=tf.float32, name=name_str))

#根据输出节点的数量返回初始化好的指定好名称的偏置
def BiasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out], dtype=tf.float32, name=name_str))

#构建编码器
def Encoder(x_origin, activate_func=tf.nn.sigmoid):
    #编码器第一隐藏层
    with tf.name_scope('layer'):
        weights = WeightsVariable(n_input_units, n_hidden_units, 'weights_encode')
        biases = BiasesVariable(n_hidden_units, 'biases_encode')
        x_code = activate_func(tf.add(tf.matmul(x_origin, weights),biases))#y=wx+b得到了编码的结果
        variable_summaries(weights)
        variable_summaries(biases)#第一步：调用汇总函数，汇总weights和biases
    return x_code

#构建解码器
def Decoder(x_code, activate_func=tf.nn.sigmoid):
    # 解码器第一隐藏层
    with tf.name_scope('layer'):
        weights = WeightsVariable(n_hidden_units, n_output_units, 'weights_decode')
        biases = BiasesVariable(n_output_units, "biases_decode")
        x_decode = activate_func(tf.add(tf.matmul(x_code, weights), biases))  # y=wx+b得到了解码的结果
        variable_summaries(weights)
        variable_summaries(biases)  # 第一步：调用汇总函数，汇总weights和biases
    return x_decode #使得x_decode特征码数据与x_original源数据 误差尽可能的小

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

    #第一步：为计算图添加损失节点的汇总标量scalar summary和 summary
    with tf.name_scope('SummaryLoss'):
        tf.summary.scalar('loss', Loss)#Loss节点
        tf.summary.scalar('learningRate', learning_rate)#learning_rate标量
    with tf.name_scope('SummaryImage'):
        image_original = tf.reshape(X_Origin, [-1, 28, 28, 1])#-1表示样本数量未知， X_Origin节点图像
        image_reconstructed = tf.reshape(X_decode, [-1, 28, 28, 1])#784维度--->28*28图片  X_decode节点图像
        tf.summary.image('image_original', image_original, 10)#最多放10张图像
        tf.summary.image('image_reconstructed', image_reconstructed, 10)

    #第二步：收集汇总节点
    merged_summaries = tf.summary.merge_all()

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

                #第三步：调用sess.run()方法，运行汇总节点，更新事件文件
                summary_str = sess.run(merged_summaries, feed_dict={X_Origin: batch_xs})#汇总的时候需要计算一遍，所以要喂数据
                summary_write.add_summary(summary_str, epoch)#横坐标为epoch，纵坐标为loss 和learning_rate               summary_write.flush()

        #关闭summary_writer,不占用资源
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