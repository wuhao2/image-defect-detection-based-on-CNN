#_*_ coding=utf-8 _*_
#%% Evaluate one image
# when training, comment the following codes.
import tensorflow as tf
import numpy as np
import model
import input_data

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)#任意选择一张图片，进行测试
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image #获取一张图片

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    train_dir = './data/train/train/'
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)#任意选择一张图片

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image) #图片标准化
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)  #因为最后一层没有激活函数，所在此处应该加上激活函数

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])#利用placeholder方式喂给数据

        # you need to change the directories to yours.
        logs_train_dir = './logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir) #读取模型结构和参数
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            #模型已经准备就绪，准备预测图片的类型
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)#得到两个概率，取最大的概率
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0]) #猫
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1]) #狗


#%%



"""
从一个文件夹中读取文件，分别判断是猫还是狗，是猫就加1，是狗也加1，然后记录准确率----效率低
"""

