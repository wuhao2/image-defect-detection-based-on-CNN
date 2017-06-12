# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/17 17:14 '

import os
import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


#过滤警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 全局变量，用来保存模型的基本参数
FLAGS = None

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.展开成784维的特征向量
IMAGE_SIZE = 28
#每个样本批次的数量
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# 创建一个带有合适初始值的weight，不能初始化为0，否则网络会卡住
# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#创建一个带有合适初始值的偏置bias变量
def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义一个张量汇总函数（均值，标准差，最大最小值，直方图）
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

#可重用代码创建一个简单的神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    做了一些矩阵乘法，偏置加法，然后使用rectify linear unit进行非线性输出映射
    也设置了一些名称域，使得产生的计算图有更好的可读性，添加了一些汇总节点
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases#每一个批次样本的每个神经元上的激活值
        tf.summary.histogram('pre_activations', preactivate)

      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations


#创建日志文件夹，启动训练过程
def training():

    #输入占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y-input')
    #reshape一下图片的形状，用于在tensorboard中显示image
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)#汇总图像节点，最多放10张图片

    #第一个隐藏层
    # hidden1  = nn_layer(x, 784, 128, 'layer1')
    hidden1 = nn_layer(x, IMAGE_PIXELS, FLAGS.hidden1, 'layer1')#都使用宏定义

    #dropout层
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)# 汇总dropout的概率
        dropped = tf.nn.dropout(hidden1, keep_prob)

    #线性输出层
    logits  = nn_layer(dropped, FLAGS.hidden1, NUM_CLASSES, 'layer2', act=tf.identity)

    #定义损失节点
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)#汇总cross_entropy

    #定义训练节点train_step
    with tf.name_scope('train'):
        #Adam自适应优化器节点
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

    #定义评估节点
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)#汇总准确率

    # 写入计算图
    graph_writer = tf.summary.FileWriter('logs/mnist_with_summaries/graph', tf.get_default_graph())
    graph_writer.flush()

    # 导入Mnist数据集
    mnist = input_data.read_data_sets(FLAGS.input_data_dir, one_hot=True, fake_data=FLAGS.fake_data)

    #生成Tensorflow feed_dict函数: 将真实的张量数据映射到占位符Tensor placeholders
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0#测试时是不会丢弃权重和偏置的
        return {x: xs, y_: ys, keep_prob: k}#填充3个占位符

    # 声明一个交互式会话
    sess = tf.InteractiveSession()
    # 初始化所有变量
    tf.global_variables_initializer().run()

    # 整合所有的汇总节点，将汇总事件文件写入logs/mnist_with_summaries中
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)  #写入训练日志，同时生成计算图
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test',sess.graph)  #写入测试日志,不生成计算图

    #一切准备就绪，启动会话，开始训练，每一批数据输出交叉熵损失
    for step in range(FLAGS.max_steps):
        # _表示第一个参数train_step不需要接收
        _, train_summary_str, XetropyLoss = sess.run([train_step, merged, cross_entropy], feed_dict(True))#True表示取训练集样本
        train_writer.add_summary(train_summary_str, global_step=step)#训练的汇总写好了
        print('step idx:', step, 'Xetropy_loss:', XetropyLoss)
        #每100步输出准确率
        if (step%100 == 0):
            acc, test_summary_str = sess.run([accuracy, merged], feed_dict=feed_dict(False))#取测试集样本
            test_writer.add_summary(test_summary_str, global_step=step)#测试的汇总做好了
            print("~~~~~~~~~~~~~~~~~~accuracy at step %s: %s~~~~~~~~~~~~~~~~" %(step, acc))
        # 在训练数据时执行训练节点，记录训练汇总信息Record train set summaries, and train
        else:
            if step % 100 == 99:  # 每100步，跟踪一次，并记录一次执行记录
                #声明跟踪选项run_options和跟踪元数据run_metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#fully_trace表示全部跟踪
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)#运行时的 内存或者运算时间。。。都放在这个里面
                #将跟踪元数据写入跟踪文件
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, step)
    #写入汇总图
    test_writer.flush()
    train_writer.flush()





def main(_):
  #创建存放事件日志和模型检查点的文件夹
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  #启动训练
  training()

#用Argument Paserar类把模型的参数全部解析到全局变量FLag中
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',#定义学习率
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--dropout',
      type=float,
      default=0.9,#丢弃90%
      help='Keep probability for training dropout.')
  parser.add_argument(
      '--max_steps',#定义最大循环次数
      type=int,
      default=1000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',#每一批次数据集样本数
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='mnist_data/',#下载数据集的目录
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='logs/mnist_with_summaries',#此目录专门用来方法event日志文件
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
