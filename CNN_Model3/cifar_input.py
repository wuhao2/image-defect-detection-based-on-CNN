
import os
from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

print("开始调用我了。。。。。")

######################################################################################################################
def read_cifar10(filename_queue):
    """
    Reads and parses examples from cifar10_apply_own_dataset data files.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    Args:
         filename_queue: A queue of strings with the filenames to read from.
    Returns:
         An object representing a single example, with the following fields:
         height: number of rows in the result (32)
         width: number of columns in the result (32)
         depth: number of color channels in the result (3)
         key: a scalar string Tensor describing the filename & record number
           for this example.
         label: an int32 Tensor with the label in the range 0..9.
         uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0) # 头字节，尾字节
    result.key, value = reader.read(filename_queue)  # 返回字节位置key和字节码value

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)  # 将字节码解码成uint8的数字向量

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        # label_bytes=1，切片得到第0个位置即label（uint8）----tf.int32
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],#[1]
                         [label_bytes + image_bytes]),#返回一个3072个长度的图像数据
        [result.depth, result.height, result.width])  #深度优先
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result  # 返回一个三通道的图像数据[height, width, depth]

######################################################################################################################

def read_cifar100(filename_queue, coarse_or_fine='fine'):
    class CIFAR100Record(object):
        pass
    result = CIFAR100Record()
    result.height = 32
    result.width = 32
    result.depth = 3
    # cifar100中每个样本都有两个类别标签，第一个字节是初略分类标签，第二个是精细类别标签
    coarse_label_bytes = 1
    fine_label_bytes = 1
    image_bytes =result.height * result.width * result.depth
    # 每一条记录都是由 标签 + 图像 组成，其字节数是固定的
    record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes
    # 创建一个固定长度记录读取器，读取一个样本记录的所有字节
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)
    # 返回一天记录
    result.key, value = reader.read(filename_queue)  # 读取test.bin和train.bin
    # 将一系列字节组成的string类型的记录----->长度为record_bytes，类型为uint8类型的数字向量
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 第一个字节为粗略标签，第二个字节为精细标签
    coarse_label = tf.cast(tf.strided_slice(record_bytes,[0], [coarse_label_bytes]), tf.int32)
    fine_label = tf.cast(tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes+fine_label_bytes]), tf.int32)

    if  coarse_or_fine == 'fine':  # 确定到底是把那个标签赋值给result_label
        result.label = fine_label
    else:
        result.label = coarse_label
    # 剩余字节都为图像数据，将其从一维张量[depth * height * width]------> 三维张量[depth ,heigth ,width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [coarse_label_bytes+fine_label_bytes],
                                              [coarse_label_bytes+fine_label_bytes+image_bytes]),
                                              [result.depth, result.height, result.width] )
    # 把图像空间位置和深度位置顺序转换  [depth ,heigth ,width]  ------> [heigth ,width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result



##########################################################################################################################

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
                            in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    #一次读取一张图片，不断的填充，直到产生一个batch批次
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16  # 预处理线程，并发执行
    # 构造训练的batch
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # 汇总图像,不在此处汇总了
    # tf.summary.image('images', images, max_outputs=9) #version 1.0的API,

    return images, tf.reshape(label_batch, [batch_size])


##################################################################################################################
# 从训练集上，产生批次数据，并进行了数据的增强
def distorted_inputs(cifar10or20or100, data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.
    Args:
      cifar10or20or100: specify which dataset cifar10 ? fine cifar100 ? coarse cifar100 ?
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if cifar10or20or100 == 10 :
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' )%i for i in xrange(1, 6)]
        read_cifar = read_cifar10
        coarse_or_fine = None
    if cifar10or20or100 == 20 :
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_or_fine = 'coarse'#初分类
    if cifar10or20or100 == 100 :
        filenames = [os.path.join(data_dir, 'train.bin')]
        read_cifar = read_cifar100
        coarse_or_fine = 'fine' #细分类

    # 检查文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)   # 产生一个文件读取队列
    # read_input = read_cifar(filename_queue, coarse_or_fine)   # 读取文件名队列，返回一个result对象
    read_input = read_cifar(filename_queue)   # 读取文件名队列，返回一个result对象
    cast_image = tf.cast(read_input.uint8image, tf.float32)  # 位深度变成浮点数uint8image--->float32

    height = IMAGE_SIZE   # 要生成的目标文件大小32*32
    width = IMAGE_SIZE

    # 为图像添加padding=4， 图像尺寸变为[32+4, 32+4]，为后面的随机裁剪留出位置
    padding_image = tf.random_crop(cast_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(padding_image)  # 随机左右水平翻转
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)  # 变换图像亮度，给每个像素点加上一个常量
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)  # 随机改变图像的对比度，给每个像素点乘以一个常量
    float_image = tf.image.per_image_standardization(distorted_image)  # 图像的标准化，0均值，unit norm

    float_image.set_shape([height, width, 3])  # Set the shapes of tensors.
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4  # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)  # 50000*0.4 = 20000 训练样本
    print('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)  # 产生样本队列min_queue_examples 20000

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)   # 从20000个样本队列中随机产生batch_size=100的样本


###############################################################################################################
# 从训练集上，产生批次数据，没有进行了数据的增强
def inputs(cifar10or20or100, eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      cifar10or20or100: specify which dataset cifar10 ? fine cifar100 ? coarse cifar100 ?
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if cifar10or20or100 == 10 :
        read_cifar = read_cifar10
        coarse_or_fine = None
        if not eval_data:  # 如果不想读取测试集
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)for i in xrange(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:  # 读取测试集
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    if cifar10or20or100 == 20 or cifar10or20or100== 100 :
        read_cifar = read_cifar100
        coarse_or_fine = None
        if not eval_data:  # 如果不想读取测试集
            filenames = [os.path.join(data_dir, 'train.bin' % i)for i in xrange(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN #50000
        else:  # 读取测试集
            filenames = [os.path.join(data_dir, 'train.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL #10000
    if cifar10or20or100 == 100:  # 如果等于100则为细分类
        coarse_or_fine = 'fine'
    if cifar10or20or100 == 20:  # 如果等于20则为粗分类
        coarse_or_fine = 'coarse'


    # 检查文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)  # 传入文件名列表 到 字符串输入生成器 产生文件名队列
    # 从文件名队列中读取样本.
    # read_input = read_cifar(filename_queue, coarse_or_fine)  # 返回的是一个类对象result
    read_input = read_cifar(filename_queue)  # 返回的是一个类对象result
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 8位图像--->32位float图像

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)  # pad进行扩充，crop裁剪3

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)  # 图像标准化

    # Set the shapes of tensors.
    # 设置数据集中张量的形状
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)  # train data ：50000*0.4=20000；test data：10000*0.4=4000

    # Generate a batch of images and labels by building up a queue of examples.
    # 产生一个批次的样本和标签
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)   # 不是随机得到batch_size=100