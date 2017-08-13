# _*_ coding: utf-8 _*_
__author__ = 'wuhao'
__date__ = '2017/8/10 13:18'

"""
# 《TensorFlow实战Google深度学习框架》07 图像数据处理
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts07.04.py # 队列操作

import tensorflow as tf

# 1. 创建队列，并操作里面的元素
q = tf.FIFOQueue(2, "int32")
init = q.enqueue_many(([0, 10],))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(8):
        v, _ = sess.run([x, q_inc])
        print(v)
'''
0
10
1
11
2
'''
"""



"""
# 2. 这个程序每隔1秒判断是否需要停止并打印自己的ID
import tensorflow as tf
import numpy as np
import threading
import time

# 线程函数
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n" % worker_id, coord.request_stop())
        else:
            print("Working on id: %d\n" % worker_id, time.sleep(1))

# 3. 创建、启动并退出线程
coord = tf.train.Coordinator()
# 声明5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
# 启动5个线程
for t in threads: t.start()
# 等待所有的线程退出
coord.join(threads)
'''
Stoping from id: 1
 None
Working on id: 0
 None
'''
"""

# 《TensorFlow实战Google深度学习框架》07 图像数据处理
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts07.05.py # 多线程队列操作

import tensorflow as tf

# 1. 定义队列及其操作
queue = tf.FIFOQueue(100, "float")
enqueue_op = queue.enqueue([tf.random_normal([1])])
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()

# 2. 启动线程
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
'''
-0.0838374
1.52686
-0.267706
'''