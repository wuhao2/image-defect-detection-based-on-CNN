# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/17 17:02 '

import math
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.展开成784维的特征向量
IMAGE_SIZE = 28
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

hidden1_units = 20
hidden2_units = 25
hidden3_units = 15
hidden4_units = 10

#实例化一个占位符
images = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name='x_input')


#可重用代码创建一个简单的神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        # variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        # variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        # tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      # tf.summary.histogram('activations', activations)
      return activations

hidden1 = nn_layer(images, IMAGE_PIXELS, hidden1_units, 'layer1')
hidden2 = nn_layer(hidden1, hidden1_units, hidden2_units, 'layer2')
hidden3 = nn_layer(hidden2, hidden2_units, hidden3_units, 'layer3')
hidden4 = nn_layer(hidden3, hidden3_units, hidden4_units, 'layer4')

writer = tf.summary.FileWriter("logs/nn_layer", tf.get_default_graph())
writer.flush()