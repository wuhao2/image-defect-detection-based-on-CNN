# _*_ coding=utf-8 _*_
import training
import input_data
import evaluate_any_image

#测试自己制作的数据集
# input_data.test_input_data()

#开始训练
training.run_training()

#利用模型测试任意一张图片: 0代表猫， 1代表狗
evaluate_any_image.evaluate_one_image()