# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/16 11:28 '

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用常量作为计算图的输入
# 创建一个常量操作constant op， op会被作为一个节点node添加到默认的计算图上
a = tf.constant(2)
b = tf.constant(3)

# 启动默认的计算图
with tf.Session() as sess:
    print("a=2, b=3")
    print("常量节点相加：%i" % sess.run(a + b))  # 直接在此处定义常量操作的方法
    print("常量节点相乘：%i" % sess.run(a * b))

# 使用变量（variable）作为计算图的输出
# 构造函数返回值就是variable op的输出（session运行时，为session提供输入
# tf Graph input
a = tf.placeholder(tf.int16)  # 占位符变量
b = tf.placeholder(tf.int16)

# 定义一些基本操作变量的方法
add = tf.add(a, b)
mul = tf.multiply(a, b)

# 启动默认会话
with tf.Session() as sess:  # 运用with as 结构 限制了session作用域
    print("变量节点相加：%i" % sess.run(add, feed_dict={a: 2, b: 3}))  # key-value方式喂数据
    print("变量节点相乘：%i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# 矩阵相乘（Matrix Multiplication）
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)

# 为了运行matmul op，条用session的run（）方法，传入’product‘
# ‘product‘表达matmul op的输出， 这表明了我们想要取回（fetch back）matmul op 输出
# op需要的所有输入都会由session自动运行，有些过程可以自动并行执行
# 调用run（product）就会引起计算图上三个节点的执行：2个constants节点和一个matmul节点
# ‘product’ op的输出返回给result即numpy的ndarray对象
with tf.Session() as sess:
    result = sess.run(product)
    print('矩阵相乘的结果：', result)  # [[12]]

writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
writer.flush()  # 不管filewrite队列中有没有满，都强制写文件，然后把队列清空
