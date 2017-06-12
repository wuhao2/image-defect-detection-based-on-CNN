# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 16:49 '
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#产生训练数据
train_X = np.asarray([3.3, 4.4, 5.5, 6.93, 4.168, 9.77, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.645, 9.27, 6.1])
train_Y = np.asarray([3.7, 2.76, 4.09, 3.19, 1.694, 4.575, 3.366, 2.596, 3.53, 6.221,
                      5.876, 3.654, 5.65, 2.094, 7.43, 4.96])
n_train_samples = train_X.shape[0]
print("训练样本数量：", n_train_samples)

#产生测试样本
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([5.84, 6.272, 3.2, 3.831, 5.92, 4.24, 8.35, 3.03])
n_test_samples = test_X.shape[0]
print("测试样本数量：", n_test_samples)

#绘制散点图，展示原始数据
plt.plot(train_X, train_Y, 'ro', label='Original Train Points')
plt.plot(test_X, test_Y, 'b*', label='Original Test Points')
plt.legend()
plt.show()#线程会阻塞在此


print("~~~~~~~~~~~~~~~~~~开始设计计算图~~~~~~~~~~~~~~~~~~~~~")
#Tensorflow模型将会被构建在默认的Graph上
with tf.Graph().as_default():

    #输入占位符
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, name='X')
        Y_true = tf.placeholder(tf.float32, name='Y_true')

    with tf.name_scope('inference'):
        #模型参数变量
        W = tf.Variable(tf.zeros([1]), name='weight')
        b = tf.Variable(tf.zeros([1]), name='bias')

        #inference：y = wx + b
        Y_pred = tf.add(tf.multiply(X, W), b)

    with tf.name_scope('loss'):
        #添加loss
        TrainLoss = tf.reduce_mean(tf.pow(Y_true-Y_pred, 2))/2 #pow幂运算

    with tf.name_scope('Train'):
        #optimizer：创建一个梯度下降优化器
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        #Train：定义训练节点，将梯度下降法应用于Loss
        TrainOp = Optimizer.minimize(TrainLoss)

    with tf.name_scope('input'):
        #添加评估节点
        EvalLoss = tf.reduce_mean(tf.pow(Y_true-Y_pred, 2))/2

    #Initial：添加所有的Variable类型的变量的初始化节点
    InitOp = tf.global_variables_initializer()

    #保存计算图
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    writer.close()#不管filewrite队列中有没有满，都强制写文件，然后把队列清空

    print("开始会话，运行计算图，训练模型")
    sess = tf.Session()
    sess.run(InitOp)
    #开始按批次训练，训练1000次
    for step in range(1000):
        for tx,ty in zip(train_X, train_Y):
            _,train_loss, train_w, train_b =sess.run([TrainOp, TrainLoss, W, b],
                                                 feed_dict={X: tx, Y_true: ty})#一个批次传入一个样本
                                                 # feed_dict={X:train_X, Y_true:train_Y})#一次传入的是所有的样本

        #每隔5步训练完后输出当前模型trainset的损失信息
        if (step + 1) % 5 ==0 :
            print("step:", '%04d' %(step + 1), "train_loss=", "{:.9f}" .format(train_loss),
                  "W=",train_w, "b=", train_b)
            # #每5步展示展示一个拟合曲线
            # plt.plot(train_X, train_Y, 'ro', label="Original Train Points")
            # plt.plot(test_X, test_Y, 'b*', label='Original Test Points')
            # plt.plot(train_X, train_w * train_X + train_b, label='Fitted Line')
            # plt.legend()
            # plt.show()#

         #每隔10步训练完成后输出模型testset的损失信息
        if (step + 1) % 10 ==0 :
            test_loss = sess.run(EvalLoss, feed_dict={X: test_X, Y_true:test_Y})
            print("step:", '%04d' % (step + 1), "test_loss=", "{:.9f}".format(test_loss),
                  "W=", train_w, "b=", train_b)
        #     # 展示拟合曲线
        #     plt.plot(train_X, train_Y, 'ro', label="Original Train Points")
        #     plt.plot(test_X, test_Y, 'b*', label='Original Test Points')
        #     plt.plot(train_X, train_w * train_X + train_b, label='Fitted Line')
        #     plt.legend()
        #     plt.show()


    print("训练完毕")

    W, b = sess.run([W, b])
    print("得到模型参数：", "W=", W, "b=", b)
    train_loss =sess.run(TrainLoss, feed_dict={X:train_X, Y_true:train_Y})
    print("训练集上的损失：", train_loss)
    test_loss =sess.run(EvalLoss, feed_dict={X: test_X, Y_true:test_Y})
    print("测试集上的损失：", test_loss)

    #展示拟合曲线
    plt.plot(train_X, train_Y, 'ro', label="Original Train Points")
    plt.plot(test_X,test_Y, 'b*', label='Original Test Points')
    plt.plot(train_X, W*train_X+b, label='Fitted Line')
    plt.legend()
    plt.show()

