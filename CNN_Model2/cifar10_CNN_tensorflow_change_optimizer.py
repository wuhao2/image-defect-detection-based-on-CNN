#_*_ coding=utf-8 _*_

import tarfile
# from six.moves import urllib
from urllib.request import urlretrieve,urlopen
import urllib
import csv

import tensorflow as tf
import os, sys
import numpy as np

# from pip._vendor.distlib._backport import tarfile
# from sklearn.externals.six.moves import urllib

import cifar10_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#数据集中输入图像的参数
dataset_dir= '../cifar10_dataset'
num_examples_per_epoch_for_train = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size = cifar10_input.IMAGE_SIZE
image_channel = 3
n_classes = cifar10_input.NUM_CLASSES#cifar10中类的数量

#算法超参数
training_epochs = 6
"""
改变学习率
"""
# learning_rate_init = 0.1
# learning_rate_init = 0.01
learning_rate_init = 0.001
# learning_rate_init = 0.0001
batch_size = 100
display_step = 20
conv1_kernels_num = 32 #64个卷积核，这是超参数，应该写到开头
conv2_kernels_num = 32 #64个卷积核
fc1_units_num = 192
fc2_units_num = 98
################################################################################################
#从网上下载并解压缩数据集from Alex's website，存放到data_dir指定的目录下
# def maybe_download_and_extract(data_dir):
#     dest_directory = data_dir
#     DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
#
#     if not os.path.exists(dest_directory):
#         os.makedirs(dest_directory)
#
#     filename = DATA_URL.split('/')[-1] #得到'cifar-10-binary.tar.gz'
#     filepath = os.path.join(dest_directory, filename) #得到../cifar10_dataset/cifar-10-binary.tar.gz
#
#     if not os.path.exists(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f %%' %
#                              (filename, float(count*block_size)/float(total_size) * 100.0))
#             sys.stdout.flush()
#
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#     print()
#     statinfo = os.stat(filepath)
#     print("Successfully downloaded", filename, statinfo.st_size, 'bytes.')
#
#     tarfile.open(filepath, 'r:gz').extractall(dest_directory)#解压文件

def get_distorted_train_batch(data_dir, batch_size):
    """
    :param data_dir:
    :param batch_size:
    :return: images 4D Tensor of [batch_size, image_size,image_size,3] size
             labels 1D Tensor of [batch_size] size
    """
    if not data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                   batch_size=batch_size)
    return images, labels#返回batch_size=100批次的样本


def get_undistorted_eval_batch(data_dir, eval_data, batch_size):
    """
    :param data_dir:
    :param eval_data:
    :param batch_size:
    :return: images 4D Tensor of [batch_size, image_size,image_size,3] size
             labels 1D Tensor of [batch_size] size
    """
    if not data_dir:
        raise ValueError('please supply a data_dir')

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(data_dir=data_dir,
                                          eval_data=eval_data,
                                          batch_size=batch_size)
    return images, labels#返回batch_size=100批次的样本



#初始化权重
def WeightVariable(shape, name_str, stddev=0.1):
    #截断正态分布
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

#初始化偏重
def BiasesVariable(shape, name_str, init_value=0.0):
    #常量节点初始化偏置
    initial = tf.constant(init_value,shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

#2维卷积，做特征提取，不做降采样，降采样交给pooling层做
def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu' ):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y,b)
    with tf.name_scope('act_name'):
        y = activation(y)
    return y

#2维池化层,空间降采样
def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)

#全连接层activate（wx+b）
def FullyConnection(x, W, b, activate=tf.nn.relu, act_name='relu'):#默认是非线性连接
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y
##################################################################################
#为每一层的激活输出添加汇总节点
def AddActivationSummary(x):
    """
    :param x: tensor
    :return: create a summary that measures the sparsity of activations
    """
    tf.summary.histogram('/activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))#稀疏性是个张量，0越多，越稀疏

#为所有的损失节点添加(滑动平均)标量汇总
def AddLossesSummary(losses): #传入的是[]
    #计算所有individual loss and total loss
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')#滑动平均算子
    loss_averages_op = loss_averages.apply(losses)

    #为所有的individual losses and total loss 绑定标量汇总节点
    #为所有的平滑处理过的individual losses and total loss 绑定标量汇总节点
    for loss in losses:
        #有平滑过的loss名字后面加上raw， 没有平滑处理过的loss使用原名称
        tf.summary.scalar(loss.op.name + 'raw', loss)
        tf.summary.scalar(loss.op.name + 'avg', loss_averages.average(loss))
    return loss_averages_op
###########################################################################################

def Inference(images_holder):
    with tf.name_scope("features_extract"):
        #第一个卷积层activate（conv2d + biase）
        with tf.name_scope('Conv2d_1'):
            # conv1_kernels_num = 32 #64个卷积核，这是超参数，应该写到开头
            weights = WeightVariable(shape=[5,5,image_channel,conv1_kernels_num],
                                     name_str='weights', stddev=5e-2)#0.005
            biases = BiasesVariable(shape=[conv1_kernels_num], name_str='biases', init_value=0.0)
            conv1_out = Conv2d(images_holder, weights, biases, stride=1, padding='SAME')
            #汇总
            AddActivationSummary(conv1_out)
        #第一个池化层pool2d
        with tf.name_scope('Pool2d_1'):
            pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

        #第二个卷积层activate（conv2d + biase）
        with tf.name_scope('Conv2d_2'):
            # conv2_kernels_num = 32 #64个卷积核
            weights = WeightVariable(shape=[5,5,conv1_kernels_num, conv2_kernels_num],
                                     name_str='weights', stddev=5e-2)#0.005
            biases = BiasesVariable(shape=[conv2_kernels_num], name_str='biases', init_value=0.0)
            conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')#没有传激活函数，默认为relu
            #汇总
            AddActivationSummary(conv2_out)
        #第二个池化层pool2d
        with tf.name_scope('Pool2d_2'):
            pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

        #将2维特征图转化为一维的特征向量
        with tf.name_scope('FeatsReshape'):
            features = tf.reshape(pool2_out, [batch_size, -1])#batch_size * 6*6*64
            feats_dim = features.get_shape()[1].value

    with tf.name_scope("classifer"):
        #第一个全连接层
        with tf.name_scope('FC1_nonlinear'):
            # fc1_units_num = 192
            weights = WeightVariable(shape=[feats_dim, fc1_units_num],
                                     name_str='weights', stddev=4e-2)
            biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)
            fc1_out = FullyConnection(features, weights, biases,
                                            activate=tf.nn.relu,
                                            act_name='relu'
                                      )
            #汇总
            AddActivationSummary(fc1_out)
        #第二个全连接层
        with tf.name_scope('FC2_nonlinear'):
            # fc2_units_num = 96
            weights = WeightVariable(shape=[fc1_units_num, fc2_units_num],
                                     name_str='weights', stddev=4e-2)
            biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)
            fc2_out = FullyConnection(fc1_out, weights, biases,
                                      activate=tf.nn.relu,
                                      act_name='relu'
                                      )
            AddActivationSummary(fc2_out)
        #第三个全连接层
        with tf.name_scope('FC3_linear'):
            fc3_units_num = n_classes #10
            weights = WeightVariable(shape=[fc2_units_num, fc3_units_num],
                                     name_str='weights', stddev=1.0/fc2_units_num)
            biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)
            logits = FullyConnection(fc2_out, weights, biases,
                                      activate=tf.identity,
                                      act_name='identity'
                                      )
            #汇总
            AddActivationSummary(logits)
            return logits
#########################################################################################
def TrainModel():
    """
    构造计算图
    """
    with tf.Graph().as_default():
        #计算图输入
        with tf.name_scope('input'):
            #与mnist不同的是，cifar10数据都进来的就是一个三维的图像数据24*24*24
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel], name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')#不是one-hot编码

        #前向预测Inference
        with tf.name_scope('Inference'):
            logits = Inference(images_holder)

        #定义损失层loss layer
        with tf.name_scope('Loss'):
            #不能使用tf.nn.softmax_cross_entropy_with_logits，因为不是one-hot编码
            corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_holder, logits=logits)#sparse会在内部将labels编码成one-hot类型
            cross_entropy_mean = tf.reduce_mean(corss_entropy)#平均损失
            total_loss = cross_entropy_mean
            #汇总损失节点
            average_losses = AddLossesSummary([total_loss])


        #定义优化训练层Train layer
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            #global_step为不可训练的参数,计数器
            """
            改变优化器为随机梯度下降算法
            """
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=global_step)#计算梯度和应用梯度

        #定义模型评估层evaluate layer
        with tf.name_scope('Evaluate'):
           top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

        #定义获取训练样本批次的计算节点
        with tf.name_scope('GetTrainBatch'):#训练样本需要扩充
            images_train, labels_train = get_distorted_train_batch(data_dir=dataset_dir,
                                                                   batch_size=batch_size)
        #定义获取测试样本批次的节点
        with tf.name_scope('GetTestBatch'):#测试样本不需要扩充
            images_test, labels_test = get_undistorted_eval_batch(data_dir=dataset_dir,
                                                                eval_data=True,
                                                                batch_size=batch_size )


        #添加所有汇总节点
        merged_summaries = tf.summary.merge_all()

        #添加初始化节点
        init_op = tf.global_variables_initializer()

        print('write graph into Tensorboard')
        summary_writer = tf.summary.FileWriter(logdir='graphs/')#日志目录
        summary_writer.add_graph( graph=tf.get_default_graph()) #添加计算图
        summary_writer.flush()#


        """
        会话启动之前，创建csv文件,将评估结果保存到csv文件中
        """
        results_list = list()
        results_list.append(["learning_rate", learning_rate_init,
                             'training_epochs', training_epochs,
                             'batch_size', batch_size,
                             'display_step', display_step,
                             'conv1_kernels_num', conv1_kernels_num,
                             'conv2_kernels_num', conv2_kernels_num,
                             'fc1_units_num', fc1_units_num,
                             'fc2_units_num', fc2_units_num])
        #用来画图的表头
        results_list.append(['training_step', 'train_loss', 'train_step', 'train_accuracy'])

        """
        启动会话，运行计算图
        """
        with tf.Session() as sess:
            sess.run(init_op)
            print('==>>>>>>==开始在训练集上训练模型==<<<<<<<<=====')
            total_batches = int(num_examples_per_epoch_for_train / batch_size) #500
            print("Per batch size：", batch_size)#100
            print("Train sample count per epoch:", num_examples_per_epoch_for_train)#50000
            print("total batch count per epoch:", total_batches)#500

            #启动读取数据的队列
            tf.train.start_queue_runners()#返回所有线程的列表
            #记录模型被训练的步数
            training_step = 0

            for epoch in range(training_epochs):
                #每一轮要把所有的batch都跑一遍
                for batch_idx in range(total_batches): #500个批次
                    #获取一个批次的样本，就必须要巡行这两个节点
                    images_batch, labels_batch = sess.run([images_train, labels_train])
                    _, loss_value, avg_losses = sess.run([train_op, total_loss, average_losses],#运行训练节点，和损失节点
                                             feed_dict={images_holder: images_batch,
                                                        labels_holder: labels_batch,
                                                        learning_rate: learning_rate_init})
                    training_step = sess.run(global_step)#global_step是一个节点，必须要用run()获得
                    # training_step += 1

                    #每训练display_step次，计算当前模型的损失和分类准确率
                    if training_step % display_step == 0:
                        #运行accuracy节点，计算当前批次的训练样本准确率
                        predictions = sess.run([top_K_op],#运行评估节点
                                               feed_dict={images_holder: images_batch,
                                                          labels_holder: labels_batch})
                        #当前批次上的预测正确的样本数
                        batch_accuracy = np.sum(predictions) / batch_size

                        #每隔display_step,记录训练集上的损失值和准确率
                        results_list.append([training_step, loss_value, training_step, batch_accuracy])

                        print("Train ste:" + str(training_step) +
                              ", Training Loss=" + "{:.6f}".format(loss_value) +
                              ", Training Accuracy=" +
                              '{:.5f}'.format(batch_accuracy))

                        summaries_str = sess.run(merged_summaries,
                                                 feed_dict={images_holder: images_batch,
                                                            labels_holder: labels_batch})
                        summary_writer.add_summary(summary=summaries_str, global_step=training_step)
                        summary_writer.flush()

            summary_writer.close()
            print("训练完毕！！！！！")


            print('==>>>>>>==开始在测试集上评估模型==<<<<<<<<=====')
            total_batches = int(num_examples_per_epoch_for_eval / batch_size)
            total_examples = total_batches * batch_size
            print("Per batch size：", batch_size)
            print("Test sample count per epoch:", total_examples)
            print("total batch count per epoch:", total_batches)

            correct_predicted = 0
            for test_step in range(total_batches):
                #运行测试节点计算图，获取一个批次测试数据
                images_batch, labels_batch = sess.run([images_test, labels_test])
                #运行accuracy节点，计算当前批次的测试样本的准确率
                predictions = sess.run([top_K_op],
                                       feed_dict={images_holder:images_batch,
                                                  labels_holder: labels_batch})
                #累积每个批次上预测正确的样本数
                correct_predicted += np.sum(predictions)
            accuracy_score = correct_predicted / total_examples
            print('------->Accuracy on test examples:', accuracy_score)

            #将测试集上的准确率添加到csv文件中
            results_list.append(["Accuracy on Test Example:", accuracy_score])


            #逐行将results.list列表中的内容写入evaluate_results.csv文件中
            results_file = open("RMSProp_rl_0001_evaluate_results.csv", 'w', newline='')
            csv_writer = csv.writer(results_file, dialect='excel')
            for row in results_list:
                csv_writer.writerow(row)
############################################################################################
def main(argv=None):
    # maybe_download_and_extract(dataset_dir)

    train_dir = 'graphs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    TrainModel()

if __name__ == '__main__':
    tf.app.run()
