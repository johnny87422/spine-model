import tensorflow as tf
import numpy as np
import pickle
import os
import time
import random
import cv2
import math
from matplotlib import pylab as plt

with open('img1.pickle','rb') as file:
    img1=pickle.load(file)/255
    print("open img1 finish")
    pass


with open('img2.pickle','rb') as file:
    img2=pickle.load(file)/255
    print("open img2 finish")
    pass

with open('ans1.pickle','rb') as file:
    ans1=pickle.load(file)
    ans1=ans1[:,:238]
    print("open ans1 finish")
    pass

with open('ans2.pickle','rb') as file:
    ans2=pickle.load(file)
    print("open ans2 finish")
    pass

with open('testimg.pickle','rb') as file:
    testimg=pickle.load(file)/255
    print("open testimg finish")
    pass

with open('testans.pickle','rb') as file:
    testans=pickle.load(file)
    print("open testans finish")
    pass

img1=np.reshape(img1,(1988,400,400,1))
img2=np.reshape(img2,(1500,400,400,1))
testimg=np.reshape(testimg,(439,400,400,1))

train_ans1=np.zeros((1988,238))
train_ans2=np.zeros((1500,238))
test_ans=np.zeros((439,238))

train_ans1[:,:]=ans1[:,:]
train_ans2[:,:]=ans2[:,:]
test_ans[:,:]=testans[:,:238]

def randomlist(number,data1,data2):
    center = (128/2,128/2)
    pointx=64
    pointy=64
    scale = 1.0
    randomimg = np.zeros((number,128,128,1))
    randomans = np.zeros((number,238))
    n = np.array(range(0, number))
    #np.random.shuffle(n)
    random_angle= np.zeros((number))
    randomimg[:]=data1[n]
    randomans[:]=data2[n]
    randomans1=np.zeros((number,238))
    #print(n[0:5])
    for i in range(number):
        #random_angle[i]=270
        random_angle[i]=random.uniform(0,0)
        pass
    #print(random_angle[0])
    #print(random_angle)
    #print(randomans[0])
    
    randomans=randomans*128
    #print(randomans[0])
    
    for i in range(number):
        angle9 = math.radians(random_angle[i])
        nrx=randomans[i][0::2]
        nry=randomans[i][1::2]
        
        randomans1[i][0::2]=(nrx-pointx)*math.cos(angle9) - (nry-pointy)*math.sin(angle9)+pointx
        
        randomans1[i][1::2]=(nrx-pointx)*math.sin(angle9) + (nry-pointy)*math.cos(angle9)+pointy
        pass
    
    randomimg=np.reshape(randomimg,(number,128,128))
    
    for i in range(number):
        M = cv2.getRotationMatrix2D(center, -random_angle[i], scale)
        randomimg[i] = cv2.warpAffine(randomimg[i], M, (128,128))
        pass

    randomimg=np.reshape(randomimg,(number,128,128,1))
    randomans1=randomans1/128
    return randomimg,randomans1



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

    def run(self,x,s1):
        batchsize=64
        trainoneleveltag=np.zeros((3488,238))
        testoneleveltag=np.zeros((439,238))
        for i in range(int(3488/batchsize)+1):
            if i != int(3488/batchsize):
                stant=i*batchsize
                end=(i+1)*batchsize
                img=x[stant:end]
                trainoneleveltag[stant:end]=self.sess.run(self.output,feed_dict={self.img:img,self.keep_prob1:1.0,self.keep_prob2:1.0})
                pass
            else:
                stant=i*batchsize
                img=x[stant:]
                trainoneleveltag[stant:]=self.sess.run(self.output,feed_dict={self.img:img,self.keep_prob1:1.0,self.keep_prob2:1.0})
                pass
            pass
        testoneleveltag=self.sess.run(self.output,feed_dict={self.img:s1,self.keep_prob1:1.0,self.keep_prob2:1.0})
        return trainoneleveltag,testoneleveltag

def resize_img_and_dot(number,img,dot):
    resize_img=np.zeros((number,128,128))
    dot=dot*400
    for i in range(number):
        resize_img[i]=cv2.resize(img[i], (128, 128))
        pass
    resize_img=resize_img.reshape(number,128,128,1)
    dot=dot*(128/400)
    dot=dot/128
    return resize_img,dot

total_img=np.zeros((3488,400,400,1))
total_img[0:1988]=img1[:]
img1=None
total_img[1988:]=img2[:]
img2=None
total_ans=np.zeros((3488,238))
total_ans[0:1988]=train_ans1[:]
train_ans1=None
total_ans[1988:]=train_ans2[:]
train_ans2=None

total_img,total_ans=resize_img_and_dot(3488,total_img,total_ans)
testimg,test_ans=resize_img_and_dot(439,testimg,test_ans)

#print(total_ans[0]*128)
with tf.Session() as sess:
    one_model=ImportGraph('onemodel\model-27559')
    total_img,total_ans1=randomlist(3488,total_img,total_ans)
    testimg,test_ans1=randomlist(439,testimg,test_ans)
    trainoneleveltag,testoneleveltag=one_model.run(total_img,testimg)
    
    #print(total_ans1[0]*400)
    pass
#print(total_ans1[0][:]*128)
#print(trainoneleveltag[0][:]*128)
#i=3
for i in range(50):
    #total_img=total_img.reshape(3488,128,128)
    testimg=testimg.reshape(439,128,128)
    #nrx1=trainoneleveltag[i][0::2]*128
    #nry1=trainoneleveltag[i][1::2]*128
    nrx1=testoneleveltag[i][0::2]*128
    nry1=testoneleveltag[i][1::2]*128
    #nrx2=total_ans1[i][0::2]*128
    #nry2=total_ans1[i][1::2]*128
    nrx2=test_ans1[i][0::2]*128
    nry2=test_ans1[i][1::2]*128
    #plt.imshow(total_img[i])
    plt.imshow(testimg[i])
    plt.plot(nrx1,nry1,'ro')
    plt.plot(nrx2,nry2,'bo')
    plt.show()





