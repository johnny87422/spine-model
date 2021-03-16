import tensorflow as tf
import numpy as np
import pickle
import os
import time
import random
import cv2
import math
from matplotlib import pylab as plt

with open('onetestimg.pickle','rb') as file:
    testimg=pickle.load(file)/255
    print("onetestimg finish")
    pass

testimg1=np.reshape(testimg,(1,400,400,1))

class ImportGraph():
    def __init__(self, loc):
    # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
        # Import saved model from location 'loc' into local graph
        # 從指定路徑載入模型到區域性圖中
            saver = tf.train.import_meta_graph(loc + '.meta',
            clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
            # 兩種方式來呼叫運算或者引數
            # FROM SAVED COLLECTION:
            #self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            self.img = self.graph.get_operation_by_name('img').outputs[0]
            self.keep_prob1 = self.graph.get_operation_by_name('keep_prob1').outputs[0]
            self.keep_prob2 = self.graph.get_operation_by_name('keep_prob2').outputs[0]
            self.output = tf.get_collection("output")[0]

    def run(self,s1):
        testoneleveltag=np.zeros((1,238))
        testoneleveltag=self.sess.run(self.output,feed_dict={self.img:s1,self.keep_prob1:1.0,self.keep_prob2:1.0})
        return testoneleveltag

def resize_img(number,img):
    resize_img=np.zeros((number,128,128))
    for i in range(number):
        resize_img[i]=cv2.resize(img[i], (128, 128))
        pass
    resize_img=resize_img.reshape(number,128,128,1)
    return resize_img

testimg1=resize_img(1,testimg1)

with tf.Session() as sess:
    one_model=ImportGraph('onemodel\model-27559')
    testoneleveltag=one_model.run(testimg1)
    pass


for i in range(1):
    #total_img=total_img.reshape(3488,128,128)
    testimg=testimg.reshape(1,400,400)
    #nrx1=trainoneleveltag[i][0::2]*128
    #nry1=trainoneleveltag[i][1::2]*128
    nrx1=testoneleveltag[i][0::2]*400
    nry1=testoneleveltag[i][1::2]*400
    #nrx2=total_ans1[i][0::2]*128
    #nry2=total_ans1[i][1::2]*128
    #nrx2=test_ans1[i][0::2]*128
    #nry2=test_ans1[i][1::2]*128
    #plt.imshow(total_img[i])
    plt.imshow(testimg[i])
    plt.plot(nrx1,nry1,'ro')
    #plt.plot(nrx2,nry2,'bo')
    plt.show()

