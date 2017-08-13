# coding: UTF-8
# 载入必要的模块
import cv2
import pygame
from pygame.locals import *
# pygame初始化
pygame.init()
text = u"."
# 设置字体和字号
font = pygame.font.SysFont('Microsoft YaHei', 100)
# 渲染图片，设置背景颜色和字体样式,前面的颜色是字体颜色
ftext = font.render(text, True, (65, 83, 130),(0, 0, 0))
newimg = pygame.transform.resize(ftext, (640, 480))
# 保存图片
pygame.image.save(newimg, "QRcode.jpg")#图片保存地址
