#_*_ coding=utf-8 _*_

# PYTHONIOENCODING="UTF-8"
# import importlib,sys  #解决python3编码问题
# importlib.reload(sys)

import tensorflow as tf
import numpy as np
import os
import imghdr

# train_dir = '/Users/wuhao/Pictures/data/train/'
train_dir = './data/train/train/'


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    with tf.name_scope("image_label_list"):

        cats = []
        label_cats = []
        dogs = []
        label_dogs = []
        for file in os.listdir(file_dir):#return simple file
            name = file.split(sep='.')   #return ['cat', '0', 'jpg']
            if name[0] == 'cat':
                cats.append(file_dir + file)
                label_cats.append(0)
            else:
                dogs.append(file_dir + file)
                label_dogs.append(1)
        print('There are %d cats \n There are %d dogs' %(len(cats), len(dogs)))

        image_list = np.hstack((cats, dogs))  # 水平 拼接成一个列表
        label_list = np.hstack((label_cats, label_dogs))

        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)  # 随机洗牌

        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(i) for i in label_list]
        return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):  # image and  label is a  list

    with tf.name_scope("batch_image_lable"):

        image = tf.cast(image, tf.string)  # 将python list格式，转换成tensorflow格式
        label = tf.cast(label, tf.int32)
        # generate an input queue
        input_queue = tf.train.slice_input_producer([image, label])  # image和label是分开的，所以用slice_input_producer
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)  # 解码jpg,png图片
        # image = tf.image.decode_image(image_contents, channels=3)  # 解码jpg,png,gif图片

        ######################################################
        # data argumentation should go to here  数据特征工程 #
        ######################################################

        image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  # 对图片进行（扩充和裁剪）
        # if you want to test the generated batches of images, you might want to comment the following line.
        image = tf.image.per_image_standardization(image)   # 数据标准化，，0-255的value进行减去均值 除以方差
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size= batch_size,
                                                  num_threads= 64,
                                                  capacity = capacity)  # 生成批次batch

        # you can also use shuffle_batch
        # CAPACITY = 256
        # image_batch, label_batch = tf.train.shuffle_batch([image,label],
        #                                                      batch_size=batch_size,
        #                                                      num_threads=64,
        #                                                      capacity=CAPACITY,
        #                                                      min_after_dequeue=CAPACITY-1)

        label_batch = tf.reshape(label_batch, [batch_size])  # 重新reshape一下 image_batch, label_batch
        image_batch = tf.cast(image_batch, tf.float32)

        return image_batch, label_batch

#########################################################################################%% TEST 测试一下
# To test the generated batches of images
# When training the model, DO comment注释 the following codes
import matplotlib.pyplot as plt


def test_input_data():
    BATCH_SIZE = 10
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208

    # train_dir = '/Users/wuhao/Pictures/data/train/'
    image_list, label_list = get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    # init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run(init_op)
        sess.run(tf.initialize_all_variables())
        i = 0  # 只需要跑几张图就够了
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)  # 监控queue的状态，不停的入列和出列
        try:
            while not coord.should_stop() and i<1:  # 只要没有要求停止 和 i<1进入循环

                img, label = sess.run([image_batch, label_batch])

                # just test one batch
                for j in np.arange(BATCH_SIZE):  # BATCH_SIZE=2，只显示2张图片
                    print('label: %d' %label[j])  # 打印第几张图片
                    plt.imshow(img[j,:,:,:])  # 显示图片
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

# #单元测试一下
# test_input_data()