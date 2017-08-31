#_*_ coding=utf-8 _*_
# from six.moves import urllib
import csv
import os
import numpy as np
import tensorflow as tf
import alexNet_cifar_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据文件保存目录
# logs_save_model = './logs/save_model'
logdir = './logs/alexNet_graph'  # 保存计算图 和 模型结构和参数


# 算法超参数
learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 20
conv1_kernels_num = 64  # 64个卷积核，这是超参数，应该写到开头
conv2_kernels_num = 192
conv3_kernels_num = 384
conv4_kernels_num = 256
conv5_kernels_num = 256
fc1_units_num = 4096
fc2_units_num = 4096

# 数据集中输入图像的参数
dataset_dir_cifar10 = '../cifar10_dataset/cifar-10-batches-bin/'
dataset_dir_cifar100 = '../cifar100_dataset/cifar-100-binary/'
num_examples_per_epoch_for_train = alexNet_cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN  #50000
num_examples_per_epoch_for_eval = alexNet_cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL    #10000

image_size = alexNet_cifar_input.IMAGE_SIZE #32
image_channel = alexNet_cifar_input.IMAGE_DEPTH #3
# 数据集中图像参数
# image_size = 32
# image_channel = 3
# n_classes = 1000
# num_examples_per_epoch_for_train = 1000
# num_examples_per_epoch_for_eval = 500

# 通过修改cifar10or20or100，就可以测试cifar10， cifar100， cifar20
# 或者使用假数据跑模型(cifar10or20or100 = -1)
cifar10or20or100 = 10
if cifar10or20or100 == 10:
    n_classes = alexNet_cifar_input.NUM_CLASSES_CIFAR10  # cifar10中类的数量
    dataset_dir = dataset_dir_cifar10

if cifar10or20or100 == 20:
    n_classes = alexNet_cifar_input.NUM_CLASSES_CIFAR20  # cifar100中类的数量
    dataset_dir = dataset_dir_cifar100

if cifar10or20or100 == 100:
    n_classes = alexNet_cifar_input.NUM_CLASSES_CIFAR100  # cifar100中类的数量
    dataset_dir = dataset_dir_cifar100

#############################################################################################


# 获取训练batch
def get_distorted_train_batch(data_dir = dataset_dir, batch_size = batch_size):
    """
    :param data_dir:
    :param batch_size:
    :return: images 4D Tensor of [batch_size, image_size,image_size,3] size
             labels 1D Tensor of [batch_size] size
    """
    if not data_dir:
        raise ValueError('please supply a data_dir')
    images, labels = alexNet_cifar_input.distorted_inputs(cifar10or20or100=n_classes, data_dir=data_dir,
                                                          batch_size=batch_size)
    return images, labels  # 返回batch_size=100批次的样本


# 获取测试batch
def get_undistorted_eval_batch(data_dir = dataset_dir, eval_data = True, batch_size = batch_size):
    """
    :param data_dir:
    :param eval_data:
    :param batch_size:
    :return: images 4D Tensor of [batch_size, image_size,image_size,3] size
             labels 1D Tensor of [batch_size] size
    """
    if not data_dir:
        raise ValueError('please supply a data_dir')
    images, labels = alexNet_cifar_input.inputs(cifar10or20or100=n_classes,
                                        data_dir=data_dir,
                                        eval_data=eval_data,
                                        batch_size=batch_size)
    return images, labels  # 返回batch_size=100批次的样本

"""
如果想要在imageNet数据集上训练模型，只需要完成这个两个函数，获取数据images和label节点
"""


#  生成假的训练数据，用于训练模型
def get_faked_train_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel],
                                          mean=0.0, stddev=1.0, dtype=tf.float32))  # 标准正太分布数据
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0,
                                           maxval=n_classes, dtype=tf.int32))  # 标准均匀分布0-999
    return images, labels


#  生成假数据用于模型测试
def get_faked_test_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel],
                                          mean=0.0, stddev=1.0, dtype=tf.float32))  # 标准正太分布数据
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0,
                                           maxval=n_classes, dtype=tf.int32))  # 标准均匀分布0-999
    return images, labels
###########################################################################################################


# 初始化biases
def BiasesVariable(shape, name_str, init_value=0.0):
    # 常量节点初始化偏置
    initial = tf.constant(init_value,shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


# 初始化weightd
def WeightVariable(shape, name_str, stddev=0.1):
    # 截断正态分布
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


# 2维卷积activation(con2d + bias)，做特征提取，不做降采样，降采样交给pooling层做
def Conv2d(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu' ):
    with tf.name_scope('conv2d_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y,b)
    with tf.name_scope('act_name'):
        y = activation(y)
    return y


# 2维池化层pooling ,空间降采样
def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)


# Fully_conection全连接层activate（wx+b）
def FullyConnection(x, W, b, activate=tf.nn.relu, act_name='relu'): # 默认是非线性连接
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y

###########################################################################################################


"""
前向预测：构造特征提取器和分类器
"""


def Inference(images_holder):
    # 第一个卷积层activate（conv2d + biase）
    with tf.name_scope('Conv2d_1'):
        weights = WeightVariable(shape=[5,5,image_channel, conv1_kernels_num],
                                 name_str='weights', stddev=5e-1)#0.05
        biases = BiasesVariable(shape=[conv1_kernels_num], name_str='biases', init_value=0.0)
        conv1_out = Conv2d(images_holder, weights, biases, stride=1, padding='SAME')
        # 汇总
        AddActivationSummary(conv1_out)
        print_activations(conv1_out)

    # 第一个池化层pool2d
    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')  # 池化核3*3 步长为2------采用重叠池化
        print_activations(pool1_out)
    # 第二个卷积层ativate（conv2d + biase）
    with tf.name_scope('Conv2d_2'):
        weights = WeightVariable(shape=[5,5,conv1_kernels_num, conv2_kernels_num],
                                 name_str='weights', stddev=5e-1)#0.005
        biases = BiasesVariable(shape=[conv2_kernels_num], name_str='biases', init_value=0.0)
        conv2_out = Conv2d(pool1_out, weights, biases, stride=1, padding='SAME')  # 没有传激活函数，默认为relu
        # 汇总
        AddActivationSummary(conv2_out)
        print_activations(conv2_out)

    # 第二个池化层pool2d
    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')  # 池化核3*3 步长为2----》重叠池化
        print_activations(pool2_out)

    # 第三个卷积层ativate（conv2d + biase）
    with tf.name_scope('Conv2d_3'):
        weights = WeightVariable(shape=[3,3,conv2_kernels_num, conv3_kernels_num],
                                 name_str='weights', stddev=5e-1)#0.005
        biases = BiasesVariable(shape=[conv3_kernels_num], name_str='biases', init_value=0.0)
        conv3_out = Conv2d(pool2_out, weights, biases, stride=1, padding='SAME')  # 没有传激活函数，默认为rel
        #汇总
        AddActivationSummary(conv3_out)
        print_activations(conv3_out)

    # 第4个卷积层ativate（conv2d + biase）
    with tf.name_scope('Conv2d_4'):
        weights = WeightVariable(shape=[3, 3, conv3_kernels_num, conv4_kernels_num],
                                 name_str='weights', stddev=5e-1)  # 0.005
        biases = BiasesVariable(shape=[conv4_kernels_num], name_str='biases', init_value=0.0)
        conv4_out = Conv2d(conv3_out, weights, biases, stride=1, padding='SAME')  # 没有传激活函数，默认为rel
        #汇总
        AddActivationSummary(conv4_out)
        print_activations(conv4_out)

    # 第5个卷积层ativate（conv2d + biase）
    with tf.name_scope('Conv2d_5'):
        weights = WeightVariable(shape=[3,3,conv4_kernels_num, conv5_kernels_num],
                                 name_str='weights', stddev=5e-1)  # 0.005
        biases = BiasesVariable(shape=[conv5_kernels_num], name_str='biases', init_value=0.0)
        conv5_out = Conv2d(conv4_out, weights, biases, stride=1, padding='SAME')  # 没有传激活函数，默认为rel
        # 汇总
        AddActivationSummary(conv5_out)
        print_activations(conv5_out)

    # 第3个池化层pool2d_5
    with tf.name_scope('Pool2d_5'):
        pool5_out = Pool2d(conv5_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')  # Valid表示不扩充边界13/2=6
        print_activations(pool5_out)
        # 得到一个32*6*6*256的4维张量

    # 将2维特征图转化为一维的特征向量
    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool5_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value  # 返回一维向量的长度

    # 第一个全连接层
    with tf.name_scope('FC1_nonlinear'):
        weights = WeightVariable(shape=[feats_dim, fc1_units_num],
                                 name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)
        fc1_out = FullyConnection(features, weights, biases,
                                  activate=tf.nn.relu,
                                  act_name='relu'
                                  )
        #汇总
        AddActivationSummary(fc1_out)
        print_activations(fc1_out)

    # 第二个全连接层
    with tf.name_scope('FC2_nonlinear'):
        weights = WeightVariable(shape=[fc1_units_num, fc2_units_num],
                                 name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)
        fc2_out = FullyConnection(fc1_out, weights, biases,
                                  activate=tf.nn.relu,
                                  act_name='relu'
                                  )
        # 汇总
        AddActivationSummary(fc2_out)
        print_activations(fc2_out)

    # 第三个全连接层
    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = WeightVariable(shape=[fc2_units_num, fc3_units_num],
                                 name_str='weights', stddev=1.0/fc2_units_num)
        biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)
        logits = FullyConnection(fc2_out, weights, biases,
                                 activate=tf.identity,
                                 act_name='identity'
                                 )
        # 汇总
        AddActivationSummary(logits)
        print_activations(logits)
        return logits



####################################################################################################################

"""
#为每一层的激活输出添加汇总节点
"""


def AddActivationSummary(x):
    """
    :param x: tensor
    :return: create a summary that measures the sparsity of activations
    """
    tf.summary.histogram('/activations', x)  # 汇总稀疏性和特征图的形态
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))  # 稀疏性是个张量，0越多，越稀疏
"""
#为所有的损失节点添加(滑动平均)标量汇总
"""


def AddLossesSummary(losses):  # 传入的是[]
    # 计算所有individual loss and total loss
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')#滑动平均算子
    loss_averages_op = loss_averages.apply(losses)

    # 为所有的individual losses and total loss 绑定标量汇总节点
    # 为所有的平滑处理过的individual losses and total loss 绑定标量汇总节点
    for loss in losses:
        # 有平滑过的loss名字后面加上raw， 没有平滑处理过的loss使用原名称
        tf.summary.scalar(loss.op.name + 'raw', loss)
        tf.summary.scalar(loss.op.name + 'avg', loss_averages.average(loss))
    return loss_averages_op
"""
打印出每一层张量的shape
"""


def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())
###########################################################################################################

"""
构造计算图
"""

with tf.Graph().as_default():
    # 计算图输入
    with tf.name_scope('input'):
        images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel], name='images')
        labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')
        # 不是one-hot编码，如果是one-hot编码,需要改写成[batch_size， n_classes]

    # 前向预测Inference
    with tf.name_scope('feedForward'):
        logits = Inference(images_holder)

    # 定义损失层loss layer
    with tf.name_scope('Loss'):
        # 不能使用tf.nn.softmax_cross_entropy_with_logits，因为不是one-hot编码
        corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            labels=labels_holder, logits=logits)  # sparse会在内部将labels编码成one-hot类型
        cross_entropy_mean = tf.reduce_mean(corss_entropy)  # 得到一个批次上的所有样本的平均损失
        total_loss_op = cross_entropy_mean
        average_loss_op = AddLossesSummary([total_loss_op])

    # 定义优化训练层Train layer
    with tf.name_scope('BackPropagation'):
        learning_rate = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        # global_step为不可训练的参数,计数器
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss_op, global_step=global_step)#计算梯度和应用梯度

    # 定义模型评估层evaluate layer
    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)#传入的参数是，预测得分logist 和真实label

######################################################################################################


    #定义获取训练样本批次的计算节点
    with tf.name_scope('GetTrainBatch'):#训练样本需要扩充
        if  cifar10or20or100 == -1:
            #使用假数据
            images_train, labels_train = get_faked_train_batch( batch_size=batch_size)
        else:
            #使用真数据
            images_train, labels_train = get_distorted_train_batch(data_dir=dataset_dir,
                                                                   batch_size=batch_size)
        tf.summary.image('images' ,images_train, max_outputs=8)##汇总获取样本训练图像
    #定义获取测试样本批次的节点
    with tf.name_scope('GetTestBatch'):#测试样本不需要扩充
        if cifar10or20or100 == -1:
            #使用假数据
            images_test, labels_test = get_faked_test_batch(batch_size=batch_size )
        else:
            #使用真数据
            images_test, labels_test = get_undistorted_eval_batch(data_dir=dataset_dir,
                                                                  eval_data=True,
                                                                  batch_size=batch_size )
        tf.summary.image('images' ,images_test, max_outputs=8)#获取样本测试图像


    ######################################################################################################
    # 收集所有的汇总节点
    merged_summaries = tf.summary.merge_all()

    # 添加初始化节点
    init_op = tf.global_variables_initializer()

    # 将计算图写入tensorBoard
    print('write graph into Tensorboard')
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    # 保存模型结构和参数
    # saver = tf.train.Saver()

    summary_writer.close()


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
                         'conv3_kernels_num', conv3_kernels_num,
                         'conv4_kernels_num', conv4_kernels_num,
                         'conv5_kernels_num', conv5_kernels_num,
                         'fc1_units_num', fc1_units_num,
                         'fc2_units_num', fc2_units_num])
    # 用来做的表excel表格头
    results_list.append(['training_step', 'train_loss', 'train_step', 'train_accuracy'])

    """
    启动会话，运行计算图
    """
    with tf.Session() as sess:
        sess.run(init_op)
        # 启动读取数据的队列
        tf.train.start_queue_runners()   # 返回所有线程的列表
        print('==>>>>>>==开始在训练集上训练模型==<<<<<<<<=====')
        total_batches = int(num_examples_per_epoch_for_train / batch_size)
        print("Per batch size：", batch_size)
        print("Train sample count per epoch:", num_examples_per_epoch_for_train)  # 50000
        print("total batch count per epoch:", total_batches)  # 50000/100  == 500

        # 记录模型被训练的步数
        training_step = 0
        for epoch in range(training_epochs):
            # 每一轮要把所有的batch都跑一遍
            for batch_idx in range(total_batches):  # 500个批次
                # 获取一个批次的样本，就必须要运行这两个节点
                images_batch, labels_batch = sess.run([images_train, labels_train])
                # 运行训练节点，和损失节点,平滑损失节点
                _, loss_value,avg_losses = sess.run([train_op, total_loss_op, average_loss_op],
                                         feed_dict={images_holder: images_batch,
                                                    labels_holder: labels_batch,
                                                    learning_rate: learning_rate_init})
                # global_step是一个节点，必须要用run()获得
                training_step = sess.run(global_step)
                # training_step += 1#也可以这样做

                # 每训练display_step次，计算当前模型的损失和分类准确率
                if training_step % display_step == 0:
                    # 运行accuracy节点，计算当前批次的训练样本准确率
                    predictions = sess.run([top_K_op],  # 运行评估节点，返回一个32个元素的bool列表，预测正确为true
                                           feed_dict={images_holder: images_batch,
                                                      labels_holder: labels_batch})
                    # 当前批次上的预测正确的样本数， 预测正确的样本总和/batch_size
                    batch_accuracy = np.sum(predictions) / batch_size

                    # 打印训练信息
                    print("training epoch:" + str(epoch) +
                          "，Train step:" + str(training_step) +
                          ", Training Loss=" + "{:.6f}".format(loss_value) +
                          ", Training Accuracy=" + '{:.5f}'.format(batch_accuracy))

                    # 每隔display_step,记录训练集上的损失值和准确率
                    results_list.append([training_step, loss_value, training_step, batch_accuracy])
                    # 运行汇总节点，返回一个protoBuffer字符串
                    summaries_str = sess.run(merged_summaries,
                                             feed_dict={images_holder: images_batch,
                                                        labels_holder: labels_batch})
                    # 添加到file_write队列
                    summary_writer.add_summary(summary=summaries_str, global_step=training_step)
                    summary_writer.flush()

                # 保存模型结构和参数
                # if training_step % 500 == 0 or (training_step + 1) == num_examples_per_epoch_for_train:
                #     checkpoint_path = os.path.join( logdir, 'model.ckpt')
                #     saver.save(sess, checkpoint_path, global_step=training_step)

        summary_writer.close()
        print("训练完毕！！！！！")




        """
        测试模型，没有必要测试几轮所有无需epoch
        """
        print('==>>>>>>==开始在测试集上评估模型==<<<<<<<<=====')
        total_batches = int(num_examples_per_epoch_for_eval / batch_size)
        total_examples = total_batches * batch_size
        print("Per batch size：", batch_size)
        print("Test sample count per epoch:", total_examples)
        print("total batch count per epoch:", total_batches)

        correct_predicted = 0.0
        # 把测试的训练集分成多个批次进行评估，最后求平均
        for test_step in range(total_batches):
            # 运行测试节点计算图，获取一个批次测试数据
            images_batch, labels_batch = sess.run([images_test, labels_test])
            # 运行accuracy节点，计算当前批次的测试样本的准确率
            predictions = sess.run([top_K_op],
                                   feed_dict={images_holder: images_batch,
                                              labels_holder: labels_batch})
            # 累积每个批次上预测正确的样本数
            correct_predicted += np.sum(predictions)
        # 在所有测试集上的正确率即 预测正确的测试样本数 / 总的测试样本数
        accuracy_score = correct_predicted / total_examples
        print('-------> Test Accuracy:', accuracy_score)

        # 将测试集上的准确率添加到csv文件中
        results_list.append(["Accuracy on Test Example:", accuracy_score])
        # 逐行将results.list列表中的内容写入evaluate_results.csv文件中
        results_file = open('alexNet_cifar.csv', 'w', newline='')
        csv_writer = csv.writer(results_file, dialect='excel')
        for row in results_list:
            csv_writer.writerow(row)