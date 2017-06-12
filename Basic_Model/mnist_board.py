# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/17 13:10 '
import math
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.展开成784维的特征向量
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
#每个样本批次的数量
batch_size = 50
#两个隐藏层的神经元个数
hidden1_units = 20
hidden2_units = 15
#学习率
learning_rate = 0.01

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))


#inference：前项预测定义
def inference(images, hidden1_units, hidden2_units):

  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

#Loss：损失函数定义
def loss(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


#Train：训练子图定义
def training(loss, learning_rate):

  with tf.name_scope('scalar_summary'):
      #为了保存loss的值，添加一个标量汇总
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('learning_rate', learning_rate)

  #创建一个变量来跟踪global step，优化器每执行一次，global_step就会加1
  global_step = tf.Variable(0, name="global_step", trainable=False)
  #根据给定的学习率，创建一个梯度下降优化器
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  train_op = optimizer.minimize(loss=loss, global_step=global_step)
  return train_op #返回Train节点

#Evaluate: 评估模型的预测性能
def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))#接收的是bool true or false型，输出的是int32类型的0 1 0 1


#调用函数
#logits输出张量
logits = inference(images_placeholder, hidden1_units, hidden2_units)
#每个batch的loss张量节点
batch_loss = loss(logits=logits, labels=labels_placeholder)
#实例化训练节点
train_on_batch = training(loss=batch_loss, learning_rate=learning_rate)
#准确率
correct_count = evaluation(logits=logits, labels=labels_placeholder)



#保存计算图到TensorBoard webUI中
writer = tf.summary.FileWriter(logdir='logs/mnist_board', graph=tf.get_default_graph())
writer.close()  # 不管filewrite队列中有没有满，flush强制写文件，然后把队列清空
