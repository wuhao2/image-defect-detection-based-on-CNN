#实现简单的神经网络队M你身体数据集进行分类： conv2d + activation+ pool + fc
import csv
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.01
#学习率要遵守 “稳中求进”的原则
#学习率太大----》损失曲线下降比较快，但曲线的形态在不同的随机起始状态下差别较大波纹较大
#学习率太小----》损失曲线下降比较慢，但曲线的形态在不同的随机起始状态下比较一致，波纹较小
training_epochs = 1
batch_size = 100
display_step = 10

n_input = 784
n_classes = 10

#初始化权重
def WeightVariable(shape, name_str, stddev=0.1):

    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

#初始化偏重
def BiasesVariable(shape, name_str, stddev=0.00001):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

#2维卷积，做特征提取，不做降采样，降采样交给pooling层做
def Conv2d(x, W, b, stride=1, padding='SAME'):

    with tf.name_scope('Wx_b'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y,b)
    return y

#2维池化层,空间降采样
def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')


#非线性激活s
def Activation(x, activation=tf.nn.relu, name='relu'):
    with tf.name_scope(name):
        y = activation(x)
    return y

#全连接层activate（wx+b）
def FullyConnection(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        t = activate(y)
    return y

#通用的评估函数,用来在训练集和验证集上进行评估
def EvaluateMode10nDataset(sess, images, labels):
    n_samples = images.shape[0]
    per_batch_size = batch_size #100
    loss = 0
    acc = 0
    #样本比较少时，一次性评估完毕； 否则拆分成若干个批次进行评估，主要是防止内存不够用
    if (n_samples <= per_batch_size):
        batch_count = 1
        loss, acc = sess.run([corss_entropy_loss, accuracy],
                             feed_dict= {X_origin: images,
                                         Y_true: labels,
                                         learning_rate: learning_rate_init}
                             )
    else:
        batch_count = int(n_samples / per_batch_size)#10000 / 100 = 100
        batch_start = 0
        for idx in range(batch_count):#评估100次
            batch_loss, batch_acc = sess.run([corss_entropy_loss, accuracy], #0 - 100
                                             feed_dict= {X_origin: images[batch_start:batch_start + per_batch_size, :],
                                                         Y_true: labels[batch_start:batch_start + per_batch_size, :],
                                                         learning_rate: learning_rate_init}
                                             )
            batch_start += per_batch_size #0  100 200 300 .......
            #累计所有批次上的损失和准确率
            loss += batch_loss
            acc += batch_acc

    return loss / batch_count, acc / batch_count


"""
#构造计算图
"""
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('input'):
        X_origin = tf.placeholder(tf.float32, [None, n_input], name='X_origin')
        Y_true = tf.placeholder(tf.float32, [None, n_classes], name='Y_true')#接受的是一个one-hot类标签
        X_image = tf.reshape(X_origin, [-1,28,28,1])#把图像数据从N*784的张量---->N*28*28*1的张量；做卷积运算

    #前向预测
    with tf.name_scope('Inference'):
        #第一个卷积层之后，28*28---->24*24的特征图16张
        with tf.name_scope('Conv2d'):
            conv1_kernels_num = 16 #16个卷积核
            weights = WeightVariable(shape=[5,5,1,conv1_kernels_num], name_str='weights')
            biases = BiasesVariable(shape=[conv1_kernels_num], name_str='biases')
            conv_out = Conv2d(X_image, weights, biases, stride=1, padding='VALID')

        #非线性激活层
        with tf.name_scope('Activate'):
            activate_out = Activation(conv_out, activation=tf.nn.relu, name='relu')

        ##第一个最大池化之后，24*24*16---->12*12*16
        with tf.name_scope('Pool2d'):
            pool_out = Pool2d(activate_out, pool=tf.nn.max_pool, k=2, stride=2)

        #将2维特征图转化为一维的特征向量
        with tf.name_scope('FeatsReshape'):
            features = tf.reshape(pool_out, [-1, 12*12*16])#-1表示样本个数未知

        #全连接层,最后一层，此处的参数是最多的
        with tf.name_scope('FullyConnection'):
            weights = WeightVariable(shape=[12*12*16, n_classes], name_str='weights')
            biases = BiasesVariable(shape=[n_classes], name_str='biases')
            Y_pred_logits = FullyConnection(features, weights, biases,
                                            activate=tf.identity, #恒等映射，此函数不在tf.nn下面
                                            act_name='logits')

    #定义损失层
    with tf.name_scope('Loss'):
        corss_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y_true, logits=Y_pred_logits))#此处不能用sigmoid，因为是多类单标签问题

    #定义优化训练层
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainer = optimizer.minimize(corss_entropy_loss)#计算梯度和应用梯度

    #定义模型评估层
    with tf.name_scope('Evaluate'):
        correct_pred = tf.equal(tf.argmax(Y_pred_logits, 1), tf.argmax(Y_true, 1))#返回True or False
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#将True or False 变成0和1后，得到准去率

    #添加初始化节点
    init = tf.global_variables_initializer()

    print('write graph into Tensorboard')
    summary_writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    summary_writer.close()

    """
    导入Mnist数据集
    """
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

    """
    将评估结果保存到文件
    """
    result_list = list()
    #写入参数配置,写在第一行
    result_list.append(['learning_rate', learning_rate,
                        'train_epochs', training_epochs,
                        'batch_size', batch_size,
                        'display_step', display_step])
    #learning_rate	Tensor("Train/Placeholder:0", dtype=float32)	train_epochs	1	batch_size	100	display_step	10

    #文件头，第二行
    result_list.append(['train_step', 'train_loss', 'validation_loss',
                        'train_step', 'train_accuracy', 'validation_accuracy'])

    """
    启动计算图
    """
    with tf.Session() as sess:
        sess.run(init)
        total_batches = int(mnist.train.num_examples/batch_size)#55000/100
        print('per batch size:', batch_size)#100
        print('Train sample size:', mnist.train.num_examples)#55000
        print('total batch count:', total_batches)#550

        training_step = 0#记录模型被训练的次数
        #指定训练轮数，每一轮把所有的训练样本都要训练一边
        for epoch in range(training_epochs):
            #每一轮都要把所有的batch跑一边
            for batch_idx in range(total_batches):
                #取出数据
                batch_x, batch_y = mnist.train.next_batch(batch_size)#次数batch_y已经是one-hot类型
                #运行优化器，训练节点
                sess.run(trainer, feed_dict={X_origin: batch_x,
                                             Y_true: batch_y,
                                             learning_rate: learning_rate_init})
                #没训练一次，加1；最终train_steps==training_epochs*tatal_batch
                training_step += 1


                #没训练display_step次，计算当前模型的损失和分类准确率
                if training_step % display_step == 0: #当batch_idx==10,第一次进入if，进行评估，依次进入20，30，40....

                    #计算当前模型在目前（最近）见过的display_step个batchsize的训练集上的损失和分类准确率
                    start_idx = max(0, (batch_idx - display_step) * batch_size) # 从0开始，评估已经训练过的样本
                    end_idx = batch_idx * batch_size #每次评估100*10==1000个最近见过的样本

                    train_loss, train_acc = EvaluateMode10nDataset(sess,
                            mnist.train.images[start_idx:end_idx, :],#对图片进行切片传入，防止内存溢出
                            mnist.train.labels[start_idx:end_idx, :],)
                            #如果不进行切片，可能导致训练集上的准确率和损失比验证集大都大，按理训练集上的损失应该是最小的
                            # 因为评估了没有经过训练的样本

                    print("train step:" + str(training_step) +
                          ", train loss=" + "{:.6f}".format(train_loss) +
                         ", train accuracy=" + "{:.5f}".format(train_acc))
                    #计算当前模型在验证集上的损失和分类准确率
                    validation_loss, validation_acc = EvaluateMode10nDataset(sess,
                                                                   mnist.validation.images,
                                                                   mnist.validation.labels,)

                    print("train step:" + str(training_step) +
                          ", validation loss=" + "{:.6f}".format(validation_loss) +
                          ", validation accuracy=" + "{:.5f}".format(validation_acc))

                    #将评估结果保存到文件,循环保存，总共要记录550次
                    result_list.append([training_step, train_loss, validation_loss,
                                        training_step, train_acc, validation_acc])

        print("训练完毕")


        #计算指定数量的测试集上的准确率，与损失
        test_sample_count = mnist.test.num_examples
        test_loss, test_accuracy = EvaluateMode10nDataset(sess, mnist.test.images, mnist.test.labels)
        print("testing sample count:", test_sample_count)
        print("Test loss:", test_loss)
        print("test accuracy:", test_accuracy)
        result_list.append(['test step', 'loss', test_loss, 'accuracy', test_accuracy])
        print("测试完成")


        #将评估结果保存到文件
        result_file = open('evaluate_results1.csv','w', newline='')
        csv_writer = csv.writer(result_file, dialect='excel')

        for row in result_list:
            csv_writer.writerow(row)
        print("写文件完成")


#改变滤波器核的数量k的值， 在训练集上的损失很低，在验证集上的损失特别高，
# 说明网络过拟合，过于相信训练集了，泛化性能不够
#不断的做实验，确定一个合适的k值