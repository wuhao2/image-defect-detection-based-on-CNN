# _*_ coding: utf-8 _*_
__author__ = 'wuhao'
__date__ = '2017/8/10 14:27'

# 《TensorFlow实战Google深度学习框架》07 图像数据处理
# win10 Tensorflow1.0.1 python3.5.3
# CUDA v8.0 cudnn-8.0-windows10-x64-v5.1
# filename:ts07.06.py # 输入文件队列

import tensorflow as tf


# 1. 生成文件存储样例数据
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shards = 2  # 文件夹数目
instances_per_shard = 2  # 文件夹中的样本数
for i in range(num_shards):
    filename = ('./output/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()

# 2. 读取文件
files = tf.train.match_filenames_once("./output/data.tfrecords-*")  # 正则表达式匹配
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })
# init = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()
with sess.as_default():
# with tf.Session() as sess:
    # 添加初始化节点
    # sess = tf.Session()
    # sess.run(init)
    sess.run(init_op)
    # tf.global_variables_initializer().run()

    print(sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
'''
[b'Records\\data.tfrecords-00000-of-00002'
 b'Records\\data.tfrecords-00001-of-00002']
[0, 0]
[0, 1]
[1, 0]
[1, 1]
[0, 0]
[0, 1]
'''
# # 3. 组合训练数据（Batching）
# example, label = features['i'], features['j']
# batch_size = 2
# capacity = 1000 + 3 * batch_size
# # capacity = 1000 + 3 * batch_size
# example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in range(3):
#         cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
#         print(cur_example_batch, cur_label_batch)
#     coord.request_stop()
#     coord.join(threads)
'''
[0 0] [0 1]
[1 1] [0 1]
[0 0] [0 1]
'''