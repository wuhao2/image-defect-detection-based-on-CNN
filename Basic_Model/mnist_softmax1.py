# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 22:16 '

import os
import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    print("~~~~~~~~~~~~~~~~~~开始设计计算图~~~~~~~~~~~~~~~~~~~~~")
    #Tensorflow模型将会被构建在默认的Graph上
    with tf.Graph().as_default():

        #输入占位符
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, shape=[None, 784], name= 'X')
            Y_true = tf.placeholder(tf.float32, shape=[None, 10], name='Y_true')

        #inference: 前向预测
        with tf.name_scope('inference'):
            W = tf.Variable(tf.zeros([784, 10]), name='Weight')
            b = tf.Variable(tf.zeros([10]), name='bias')
            logits = tf.add(tf.matmul(X, W), b)

            with tf.name_scope('Softmax'):
                #softmax把logits变成概率分布
                Y_pred = tf.nn.softmax(logits=logits)

        #Loss: 定义损失节点
        with tf.name_scope('Loss'):
            TrainLoss = tf.reduce_mean(
                -tf.reduce_sum(Y_true*tf.log(tf.nn.softmax(Y_pred)), axis=1))  # 交叉熵损失

        #Train：定义训练节点
        with tf.name_scope('Train'):
            # optimizer：创建一个梯度下降优化器
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
            #Train：定义训练节点，将梯度下降法应用于Loss
            TrainStep = Optimizer.minimize(TrainLoss)

        #Evaluation：定义评估节点
        with tf.name_scope('Evaluate'):
            correct_prediction = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y_true,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # Initial：添加所有的Variable类型的变量的初始化节点
        InitOp = tf.global_variables_initializer()

        # 保存计算图
        writer = tf.summary.FileWriter(logdir='logs/mnist_softmax', graph=tf.get_default_graph())
        writer.close()  # 不管filewrite队列中有没有满，都强制写文件，然后把队列清空


        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~开始运行计算图~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #导入Mnist数据集
        mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

        #声明一个交互式会话
        sess = tf.InteractiveSession()

        #初始化所有变量：W， b
        sess.run(InitOp)

        # 开始按批次训练，训练1000次
        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)#,每次取出100个样本，batch_xs样本特征向量集合，batch_ys样本标签集合
            #将当前批次的样本喂给计算图中的输入占位符，启动训练节点开始训练

            _, train_loss = sess.run([TrainStep, TrainLoss],  #想要fetch的两个参数
                            feed_dict = {X:batch_xs, Y_true:batch_ys})

            print("train step:", step, ", tarin_loss:", train_loss)

        accuracy_scor = sess.run(accuracy, feed_dict={X:mnist.test.images, Y_true:mnist.test.labels})
        print("模型准确率：", accuracy_scor)

if __name__ == '__main__':
  #实例化一个解析器
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
