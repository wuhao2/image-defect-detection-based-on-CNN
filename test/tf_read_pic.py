# import tensorflow as tf
# import numpy as np
# from PIL import Image
#
# filenames = ['/Users/wuhao/Pictures/data/train/cat*.jpg']
# filename_queue = tf.train.string_input_producer(filenames)
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
#
# my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
#
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
#
#     # Start populating the filename queue.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(1): #length of your filename list
#         image = my_img.eval() #here is your image Tensor :)
#
#     print(image.shape)
#     Image.fromarray(np.asarray(image)).show()
#
#     coord.request_stop()
#     coord.join(threads)

from scipy import misc
import tensorflow as tf

img = misc.imread('cifar10.jpg')
print (img.shape )   # (32, 32, 3)

img_tf = tf.Variable(img)
print (img_tf.get_shape().as_list())  # [32, 32, 3]


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(img_tf)


import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(im)
fig.add_subplot(1,2,2)
plt.imshow(img)
plt.show()