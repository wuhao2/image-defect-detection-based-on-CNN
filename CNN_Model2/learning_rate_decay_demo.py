import csv
import os, sys
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_epochs = 5
num_examples_per_epoch_for_train = 10000
batch_size = 100
learning_rate_init = 0.1
learning_rate_final = 0.001

learning_rate_decauy_rate = 0.8#改变这个值的大小，查看learning_rate下降曲线
# learning_rate_decauy_rate = 0.5

num_batchs_per_epoch = int(num_examples_per_epoch_for_train/batch_size)
num_epochs_per_decay = 1
learning_rate_decay_steps = int(num_batchs_per_epoch * num_epochs_per_decay)

with tf.Graph().as_default():
    #优化器调用次数计数器，全局训练步数
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    #指数衰减学习率,返回的学习率是一个variable类型的变量
    learning_rate = tf.train.exponential_decay(learning_rate_init, global_step,
                                                learning_rate_decay_steps,
                                                learning_rate_decauy_rate,
                                                staircase=False)

    # #多项式衰减学习率
    # learining_rate = tf.train.polynomial_decay(learning_rate_init, global_step,
    #                                             learning_rate_decay_steps,
    #                                             learning_rate_final,
    #                                             power=0.5, cycle=False)

    # #自然指数衰减学习率
    # learining_rate = tf.train.natural_exp_decay(learning_rate_init, global_step,
    #                                             learning_rate_decay_steps,
    #                                             learning_rate_decauy_rate,
    #                                             staircase=False)

    # #指数衰减学习率
    # learining_rate = tf.train.inverse_time_decay(learning_rate_init, global_step,
    #                                             learning_rate_decay_steps,
    #                                             learning_rate_decauy_rate,
    #                                             staircase=False)

    #定义损失函数
    weights = tf.Variable(tf.random_normal([9000,9000], mean=0.0, stddev=1e9, dtype=tf.float32))
    myloss = tf.nn.l2_loss(weights, name='lL2Loss')

    #传入learning_rate创建优化器对象
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #传入global_step传入minimize(),每次调用minimize都会使得global_step自增1
    training_op = optimizer.minimize(myloss, global_step=global_step)

    #添加所有变量的初始化节点
    init_op = tf.global_variables_initializer()

    #将评估结果保存到文件
    results_list = list()
    results_list.append(['train_step', 'training_rate', 'train_step', 'train_loss'])

    #启动会话，训练模型
    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(training_epochs):
            print('******************开始训练******************')
            for batch_idx in range(num_batchs_per_epoch):

                current_learning_rate = sess.run(learning_rate)
                _, loss_value, training_step = sess.run([training_op, myloss, global_step])

                print( "Train epoch:" + str(epoch) + '\t'
                       "Train step:" + str(training_step) +
                       ", Training Rate=" + "{:.6f}".format(current_learning_rate) +
                       ", Training Loss=" + '{:.6f}'.format(loss_value))

                #记录结果
                results_list.append([training_step, current_learning_rate, training_step, loss_value])

    #将评估结果保存到文件
    print("**************训练结束****************")
    results_file = open('lr=0.8_evaluate_results.csv', 'w', newline='')
    csv_writer = csv.writer(results_file, dialect='excel')
    for row in results_list:
        csv_writer.writerow(row)















