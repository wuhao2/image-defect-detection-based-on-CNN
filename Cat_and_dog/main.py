# _*_ coding=utf-8 _*_
import training
import input_data

#测试自己制作的数据集
input_data.test_inputdata()

#开始训练
training.run_training()

#利用模型测试任意一张图片
training.evaluate_one_image()