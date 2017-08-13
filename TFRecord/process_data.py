import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("scharr.jpg", 'rb').read()
image_raw_data1 = tf.gfile.FastGFile("sobel.jpg", 'rb').read()
image_raw_data2 = tf.gfile.FastGFile("canny.jpg", 'rb').read()
# image_raw_data = tf.gfile.FastGFile("4.jpg", 'rb').read()
# with tf.Session() as sess:
sess = tf.InteractiveSession()
with sess.as_default():
    # img_data = tf.image.decode_png(image_raw_data)
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data1 = tf.image.decode_jpeg(image_raw_data1)
    img_data2 = tf.image.decode_jpeg(image_raw_data2)

    # resized = tf.image.resize_images(img_data, [500, 500], method=1)
    # resized1 = tf.image.resize_images(img_data1, [500, 500], method=1)
    # resized2 = tf.image.resize_images(img_data2, [500, 500], method=1)
    # img_data.set_shape([350, 350, 3])
    # print(img_data.get_shape())
    # print(img_data.eval())
    plt.subplot(131)
    plt.imshow(img_data.eval())
    plt.subplot(132)
    plt.imshow(img_data1.eval())
    plt.subplot(133)
    plt.imshow(img_data2.eval())
    plt.show()

"""
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    print(img_data.get_shape())
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile("output", 'wb') as f:
    #     f.write(encoded_image.eval())

    # 3. 重新调整图片大小
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    resized1 = tf.image.resize_images(img_data, [300, 300], method=1)
    resized2 = tf.image.resize_images(img_data, [300, 300], method=2)
    resized3 = tf.image.resize_images(img_data, [300, 300], method=3)
    print(resized.get_shape())

    plt.subplot(221)
    plt.imshow(resized.eval())
    plt.subplot(222)
    plt.imshow(resized1.eval())
    plt.subplot(223)
    plt.imshow(resized2.eval())
    plt.subplot(224)
    plt.imshow(resized3.eval())
    plt.show()

"""




# # 4. 裁剪和填充图片
# with tf.Session() as sess:
#     padded = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
#     croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
#
#     plt.subplot(131)
#     plt.imshow(img_data.eval(), label="source image")
#     plt.subplot(132)
#     plt.imshow(croped.eval(), label="300x300")
#     plt.subplot(133)
#     plt.imshow(padded.eval(),label='500x500')
#     plt.show()

# # 5. 截取中间50%的图片
# with tf.Session() as sess:
#     central_cropped = tf.image.central_crop(img_data, 0.5)
#     plt.imshow(central_cropped.eval())
#     plt.show()

# 6. 翻转图片
# with tf.Session() as sess:
#
#     # 左右翻转
#     flipped1 = tf.image.flip_left_right(img_data)
#     # 上下翻转
#     flipped2 = tf.image.flip_up_down(img_data)
#     # 对角线翻转
#     transposed = tf.image.transpose_image(img_data)
#
#     # 以一定概率上下翻转图片。
#     # flipped = tf.image.random_flip_up_down(img_data)
#     # 以一定概率左右翻转图片。
#     # flipped = tf.image.random_flip_left_right(img_data)
#
#     plt.subplot(221)
#     plt.imshow(img_data.eval())
#     plt.subplot(222)
#     plt.imshow(flipped1.eval())
#     plt.subplot(223)
#     plt.imshow(flipped2.eval())
#     plt.subplot(224)
#     plt.imshow(transposed.eval())
#     plt.show()


# # 7. 图片色彩调整
# with tf.Session() as sess:
#     # 将图片的亮度-0.5。
#     adjusted = tf.image.adjust_brightness(img_data, -0.5)
#     # 将图片的亮度0.5
#     adjusted1 = tf.image.adjust_brightness(img_data, 0.2)
#     # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
#     adjusted2 = tf.image.random_brightness(img_data, max_delta=0.3)
#
#     fig = plt.figure()
#     ax0= fig.add_subplot(241)
#     plt.imshow(img_data.eval())
#     ax1 = fig.add_subplot(242)
#     plt.imshow(adjusted.eval())
#     ax2 = fig.add_subplot(243)
#     plt.imshow(adjusted1.eval())
#     ax3 = fig.add_subplot(244)
#     plt.imshow(adjusted2.eval())
#     # plt.show()
#
#     # 将图片的对比度-5
#     adjusted3 = tf.image.adjust_contrast(img_data, -5)
#     # 将图片的对比度+5
#     adjusted4 = tf.image.adjust_contrast(img_data, 5)
#     # 在[lower, upper]的范围随机调整图的对比度。
#     adjusted5 = tf.image.random_contrast(img_data, 1, 5)
#     ax= fig.add_subplot(245)
#     plt.imshow(img_data.eval())
#     ax4 = fig.add_subplot(246)
#     plt.imshow(adjusted3.eval())
#     ax5 = fig.add_subplot(247)
#     plt.imshow(adjusted4.eval())
#     ax6 = fig.add_subplot(248)
#     plt.imshow(adjusted5.eval())
#
#     plt.show()


# # 8. 添加色相和饱和度
# with tf.Session() as sess:
#     adjusted = tf.image.adjust_hue(img_data, 0.1)
#     adjusted1 = tf.image.adjust_hue(img_data, 0.3)
#     adjusted2 = tf.image.adjust_hue(img_data, 0.6)
#     adjusted3 = tf.image.adjust_hue(img_data, 0.9)
#     # # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
#     # adjusted4 = tf.image.random_hue(img_data, max_delta=0.5)
#
#
#     # 将图片的饱和度-5。
#     adjusted5 = tf.image.adjust_saturation(img_data, -5)
#     # 将图片的饱和度+5。
#     adjusted6 = tf.image.adjust_saturation(img_data, 5)
#     # 在[lower, upper]的范围随机调整图的饱和度。
#     adjusted7 = tf.image.random_saturation(img_data, 1, 5)
#
#     # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
#     # adjusted8 = tf.image.per_image_whitening(img_data)
#     adjusted8 = tf.image.per_image_standardization(img_data)   # 在python3中已经将标准化函数改变了
#
#     fig = plt.figure()
#     axi1 = fig.add_subplot(241)
#     plt.imshow(adjusted.eval())
#     axi2 = fig.add_subplot(242)
#     plt.imshow(adjusted1.eval())
#     axi3 = fig.add_subplot(243)
#     plt.imshow(adjusted2.eval())
#     axi4 = fig.add_subplot(244)
#     plt.imshow(adjusted3.eval())
#
#     axi5 = fig.add_subplot(245)
#     plt.imshow(img_data.eval())   # 原图
#     axi6 = fig.add_subplot(246)
#     plt.imshow(adjusted5.eval())
#     axi7 = fig.add_subplot(247)
#     plt.imshow(adjusted6.eval())
#     axi8 = fig.add_subplot(248)
#     plt.imshow(adjusted7.eval())
#     plt.show()



#
# # 9. 添加标注框并裁减
# with tf.Session() as sess:
#     img_data = tf.image.resize_images(img_data, [180, 267], method=1)
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
#     boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
#     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
#                 tf.shape(img_data), bounding_boxes=boxes)
#     image_with_box = tf.image.draw_bounding_boxes(batched, boxes)
#
#     distorted_image = tf.slice(img_data, begin, size)
#     plt.imshow(distorted_image.eval())
#     plt.show()

