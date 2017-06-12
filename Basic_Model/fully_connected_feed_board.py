# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 22:16 '
import os
import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import mnist_board

#过滤警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 全局变量，用来保存模型的基本参数
FLAGS = None


#定义创建占位符的函数
def placeholder_inputs(batch_size):

  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist_board.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):

  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict



#定义评估函数，传入sees， 评估接点，占位符，数据集
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  #运行一个回合的评估过程 And run one epoch of eval.
  true_count = 0  # 预测正确的样本计数Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size#每个回合的执行步数
  num_examples = steps_per_epoch * FLAGS.batch_size#样本总数

  #累加每个批次样本中预测正确的样本数量
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)

  precision = float(true_count) / num_examples

  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  #获取用于训练、测试、验证的图像数据 和 类别标签集合
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # 创建占位符，为图像特征向量和标签数据Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # 前向构建 用于预测的计算图 Build a Graph that computes predictions from the inference model.
    logits = mnist_board.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # 添加损失节点
    loss = mnist_board.loss(logits, labels_placeholder)

    # 为计算图添加计算和应用梯度的训练节点
    train_op = mnist_board.training(loss, FLAGS.learning_rate)

    # 评估准确率，比较logits和labels_placeholder
    eval_correct = mnist_board.evaluation(logits, labels_placeholder)

    # 构建汇总张量
    summary = tf.summary.merge_all()

    # 添加变量初始化节点
    init = tf.global_variables_initializer()

    #创建一个saver节点。用于写入训练过程中的模型检查点文件
    saver = tf.train.Saver()

    # 创建一个session，用来计算计算图中的节点
    sess = tf.Session()

    # 实例化一个summaryWriter，用来输出summaries和graph
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)#注意sess.grap每次调用一次，就会覆盖掉以前的日志文件
    # summary_writer.flush()

    # 所有的工作真被就绪，运行初始化节点，来初始化所有的变量
    sess.run(init)

    # 开始训练，for循环不断的运行Train节点
    for step in xrange(FLAGS.max_steps):#step就是global_step,每更新完一次weight和bias，step加1
      start_time = time.time()

      #喂给数据
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)
      #fetch取得loss值
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
      #耗费的时间
      duration = time.time() - start_time

      #Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # 每个100个批次格式化输出
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))#Step 600: loss = 0.70 (0.003 sec)
        # 更新日志文件
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # 周期性保存检查点文件 并评估当前模型的性能Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:#只保存2次
        #构造checkpoint_file
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        #保存模型参数
        saver.save(sess, checkpoint_file, global_step=step)#当前会话，检查点文件，第几步的模型参数

        #评估训练集.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        #评估验证集.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        #评估测试集.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)


#创建日志文件夹，启动训练过程
def main(_):
  #创建存放事件日志和模型检查点的文件夹
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  #启动训练
  run_training()

#用Argument Paserar类把模型的参数全部解析到全局变量FLag中
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.05,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
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
      '--batch_size',
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
      default='logs/fully_connected_feed_board',#此目录专门用来方法event日志文件
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
