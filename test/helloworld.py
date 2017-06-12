# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 11:18 '

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#创建一个常量操作constant op， op会被作为一个节点node添加到默认的计算图上

#构造函数返回值就是常量节点constant op的输出
hello = tf.constant('hello tensorflow!')

#启动tensorflow会话
sess = tf.Session()

#运行hello节点
print(sess.run(hello))