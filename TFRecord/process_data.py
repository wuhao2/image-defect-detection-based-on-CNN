import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("134.jpg", 'rb').read()

# with tf.Session() as sess:
sess = tf.InteractiveSession()
with sess.as_default():
    img_data = tf.image.decode_jpeg(image_raw_data)
    # img_data.set_shape([350, 350, 3])
    # print(img_data.get_shape())
    # print(img_data.eval())
    # plt.imshow(img_data.eval())
    # plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile("output", 'wb') as f:
    #     f.write(encoded_image.eval())

    # 3. 重新调整图片大小
    resized = tf.image.resize_images(img_data, [300, 300], method=1)
    print(resized.get_shape())
    plt.imshow(resized.eval())
    plt.show()

# 4. 裁剪和填充图片
with tf.Session() as sess:
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()

# 5. 截取中间50%的图片
with tf.Session() as sess:
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

# 6. 翻转图片
with tf.Session() as sess:
    # 上下翻转
    flipped1 = tf.image.flip_up_down(img_data)
    # 左右翻转
    flipped2 = tf.image.flip_left_right(img_data)

    # 对角线翻转
    transposed = tf.image.transpose_image(img_data)

    # 以一定概率上下翻转图片。
    # flipped = tf.image.random_flip_up_down(img_data)
    # 以一定概率左右翻转图片。
    flipped = tf.image.random_flip_left_right(img_data)
    plt.subplot(221)
    plt.imshow(flipped1.eval())
    plt.subplot(222)
    plt.imshow(flipped2.eval())
    plt.subplot(223)
    plt.imshow(transposed.eval())
    plt.subplot(224)
    plt.imshow(flipped.eval())
    plt.show()


# 7. 图片色彩调整
with tf.Session() as sess:
    # 将图片的亮度-0.5。
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)

    # 将图片的亮度-0.5
    # adjusted = tf.image.adjust_brightness(img_data, 0.5)

    # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)

    # 将图片的对比度-5
    # adjusted = tf.image.adjust_contrast(img_data, -5)

    # 将图片的对比度+5
    # adjusted = tf.image.adjust_contrast(img_data, 5)

    # 在[lower, upper]的范围随机调整图的对比度。
    # adjusted = tf.image.random_contrast(img_data, lower, upper)

    plt.imshow(adjusted.eval())
    plt.show()
