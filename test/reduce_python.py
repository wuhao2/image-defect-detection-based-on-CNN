# _*_ coding: utf-8 _*_
__author__ = 'bobby'
__data__ = '2017/5/19 9:55 '

from functools import reduce

list = [1,2,3,4,5]
print (reduce(lambda x,y: x*y,list))#返回120
print (reduce(lambda x,y: x+y,list))#返回15
