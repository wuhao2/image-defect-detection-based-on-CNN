# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/18 16:07 '
"""
降噪自动编码器
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Xavier均匀初始化
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#加速高斯噪声的自动编码器
class AdditiveGaussianNoiseAutoEncoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function#激活函数
        self.scale = tf.placeholder(tf.float32)
        self.training_sacle =scale#噪声的标准差
        # network_weights = self._initialize_weights()
        # self.weights = network_weights
        self.weights = dict()#所有的权重和偏置都放在这个字典中

        # 输入层
        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
        #添加噪声层
        with tf.name_scope('NoiseAdder'):
            self.scale = tf.placeholder(tf.float32)
            self.noise_x = self.x + scale*tf.random_normal((n_input,))#添加噪声

        # 编码器层
        with tf.name_scope('Encoder'):
            #隐藏层的输出
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')#初始化隐藏层的权重和偏置
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias1')
            self.hidden = self.transfer_function(tf.add(tf.matmul(self.noise_x, self.weights['w1']),
                                                        self.weights['b1']))
        # 重构层
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32), name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='bias2')#初始化隐藏层的权重和偏置
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),
                                     self.weights['b2'])
        with tf.name_scope('Loss'):
            self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))
        # 训练层
        with tf.name_scope('Train'):
            self.optimizer = optimizer.minimize(self.cost)


        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print("begin to run session")


    #初始化所有variable变量
    # def _initialize_weights(self):
    #     all_weights = dict()
    #     all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')
    #     all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias1')
    #     all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32), name='weight2')
    #     all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],  dtype=tf.float32), name='bias2')
    #     return all_weights

    #一个批次傻瓜训练模型

    def partial_fit(self,X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x:X, self.scale:self.training_sacle})
        return cost

    #在给定的样本集合上计算损失（用于测试阶段）
    def calc_total_cost(self, X):
        return self.sess.run(self.cost,#表示运行cost节点，进行评估测试
                                 feed_dict={self.x: X, self.scale: self.training_sacle})#喂数据

    #返回自编码器隐藏层的输出结果，获得抽象后的高阶特征表示
    def transform(self, X):
        return self.sess.run(self.hidden,
                                 feed_dict={self.x:X, self.scale:self.training_sacle})

    #将隐藏层的高阶特征作为输入，将其重建为原始输入数据
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    #整体运行一遍复原过程，包括提取高阶特征以及重构原始数据，输入原始数据，输出复原后的数据
    def reconstruction(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X, self.scale: self.training_sacle})

    #获取隐藏层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #获取隐藏层偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

############################################################################################################
#实例化类对象
AGN_AutoEncoder = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                          scale=0.1)


print("write calculation graph into eventfile, show on tensorboard")
writer = tf.summary.FileWriter(logdir='logs', graph=AGN_AutoEncoder.sess.graph)
writer.close()

#加载数据集
mnist = input_data.read_data_sets('../Basic_Model/mnist_data/', one_hot=True)


#使用sklearn.preprocess的数据标准化操作，（0均值，标准差为1)预处理数据
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)#fit()在训练集上估计均值与方差
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)#直接用训练集估计出来的均值与方差
    return X_train, X_test


#获取随机block数据的函数：娶一个从0-len(data)的batch_size的随机整数
#以这个随机整数为随机索引，抽出一个batch_size批次的样本
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)- batch_size)#返回一个随机的小于len(data)- batch_size的数（索引）
    return data[start_index:(start_index + batch_size)]

#使用标准化操作变换数据集
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

#定义训练参数
n_samples = int(mnist.train.num_examples)#训练样本的总数
training_epochs = 20#最大训练回合数，n_samples/batch_size为1个回合
batch_size = 128#每一批次的样本数量
display_step = 1#输出训练结果的间隔

#开始训练过程：每一回合epoch训练开始，将平均损失设为0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)#使用又放回抽样，不能保证每个样本都被抽到并参与训练
    #在每个batch的循环中，随机抽取一个batch的数据，使用成员函数partial_fit，训练个batch
    #的数据，计算cost，累积到当前回合的平均cost中
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = AGN_AutoEncoder.partial_fit(batch_xs)
        avg_cost += cost/batch_size
    avg_cost /= total_batch#得到平均损失

    if epoch % display_step == 0:
        print("epoch : %04d, cost = %.9f" %(epoch+1, avg_cost))

#计算测试集上的cost
print('total cost:', str(AGN_AutoEncoder.calc_total_cost(X_test)))