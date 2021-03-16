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

#print(img1.shape)
#print(img2.shape)
#print(ans1.shape)
#print(ans2.shape)
#print(testimg.shape)
#print(testans.shape)


img1=np.reshape(img1,(1988,400,400,1))
img2=np.reshape(img2,(1500,400,400,1))
testimg=np.reshape(testimg,(439,400,400,1))

train_ans1=np.zeros((1988,238))
train_ans2=np.zeros((1500,238))
test_ans=np.zeros((439,238))

train_ans1[:,:]=ans1[:,:]
train_ans2[:,:]=ans2[:,:]
test_ans[:,:]=testans[:,:238]

img = tf.placeholder(tf.float32,[None,128,128,1], name="img")
y = tf.placeholder(tf.float32, [None,238], name="y")
keep_prob1 = tf.placeholder(tf.float32, name="keep_prob1")
keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")

def variable(name,x1,inputsize,outputsize):
    a=tf.get_variable(name,[x1,x1,inputsize,outputsize],initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True,factor=1.0),regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
    return a

def one_cnn(inputdata,wight1,strides1):
    x = tf.nn.conv2d(inputdata,wight1,strides=strides1,padding="SAME")
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    #x = tf.nn.dropout(x,keep_prob)
    return x

def cnn(inputdata,wight1,wight2,strides1,strides2,keep_prob):
    x = tf.nn.conv2d(inputdata,wight1,strides=strides1,padding="SAME")
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    x = tf.nn.dropout(x,keep_prob)
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.conv2d(x,wight2,strides=strides2,padding="SAME")
    x = tf.nn.relu(x)
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    x = tf.nn.dropout(x,keep_prob)
    return x

def dence(inputdata,output,keep_prob):
    x= tf.contrib.layers.fully_connected(inputdata,num_outputs=output,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    x= tf.nn.dropout(x,keep_prob)
    return x

def outputdence(inputdata,output,keep_prob):
    x= tf.contrib.layers.fully_connected(inputdata,num_outputs=output,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    x= tf.nn.dropout(x,keep_prob)
    return x

def randomlist(number,data1,data2):
    center = (128/2,128/2)
    pointx=64
    pointy=64
    scale = 1.0
    randomimg = np.zeros((number,128,128,1))
    randomans = np.zeros((number,238))
    n = np.array(range(0, number))
    np.random.shuffle(n)
    random_angle= np.zeros((number))
    randomimg[:]=data1[n]
    randomans[:]=data2[n]
    randomans1=np.zeros((number,238))
    #print(n[0:5])
    for i in range(number):
        #random_angle[i]=270
        random_angle[i]=random.randrange(0,360)
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
        

w1 = variable("w1",5,1,64)
w2 = variable("w2",5,64,128)
w3 = variable("w3",3,128,256)
w4 = variable("w4",3,256,512)
w5 = variable("w5",3,512,512)
#w6 = variable("w6",3,256,256)

c1 = cnn(img,w1,w2,[1,1,1,1],[1,1,1,1],keep_prob1)
c1=one_cnn(c1,w3,[1,1,1,1])
c1=one_cnn(c1,w4,[1,1,1,1])
c1=one_cnn(c1,w5,[1,1,1,1])
#c1 = cnn(c1,w3,w4,[1,2,2,1],[1,2,2,1],keep_prob1)
#c1 = cnn(c1,w5,w6,[1,2,2,1],[1,2,2,1],keep_prob1)
c1 = tf.layers.flatten(c1)
c1 = dence(c1,1024,keep_prob2)
c1 = dence(c1,1024,keep_prob2)
c1 = dence(c1,1024,keep_prob2)
c1 = dence(c1,512,keep_prob2)
#c1 = dence(c1,64,keep_prob2)
#c1 = dence(c1,256,keep_prob2)
#c1 = dence(c1,64,keep_prob2)
c1 = outputdence(c1,238,1)

tf.add_to_collection('output', c1)

mse_loss = tf.reduce_mean(tf.square(c1-y))
abs_loss = tf.reduce_mean(tf.abs(c1-y))

learning_rate_number=0.0001
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_number).minimize(mse_loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_number).minimize(mse_loss)

saver = tf.train.Saver()

batchsize = 32

picture1 = np.zeros((batchsize,128,128,1))
picture2 = np.zeros((3488-int(3488/batchsize)*batchsize,128,128,1))
#picture3 = np.zeros((1500-int(1500/batchsize)*batchsize,400,400,1))
picture_ans1 = np.zeros((batchsize,238))
picture_ans2 = np.zeros((3488-int(3488/batchsize)*batchsize,238))
#picture_ans3 = np.zeros((1500-int(1500/batchsize)*batchsize,238))
'''
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
total_img,total_ans=randomlist(3488,total_img,total_ans)

for i in range(5):
    total_img=total_img.reshape(3488,128,128)
    plt.imshow(total_img[i])
    plt.plot(total_ans[i][0::2]*128,total_ans[i][1::2]*128,'bo')
    plt.show()
'''

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)    
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
    
    for each in range(100001):
        all_loss1=0
        #all_loss2=0
        start = time.time()
        print("each="+str(each))
        total_img1,total_ans1=randomlist(3488,total_img,total_ans)
        #print("randomlist-finish")

        for i in range(int(3488/batchsize)):
            
            #if i != int(3488/batchsize):
            stant=i*batchsize
            end=(i+1)*batchsize
            picture1[:]=total_img1[stant:end]
            picture_ans1[:]=total_ans1[stant:end]
            _=sess.run([optimizer],feed_dict={img:picture1,y:picture_ans1,keep_prob1:0.8,keep_prob2:0.8})
            #pass
            
            '''else:
                stant=i*batchsize
                picture2[:]=total_img[stant:]
                #print(np.isnan(picture2))
                picture_ans2[:]=total_ans[stant:]
                #print(np.isnan(picture_ans2))
                _=sess.run([optimizer],feed_dict={img:picture2,y:picture_ans2,keep_prob1:0.8,keep_prob2:0.5})
                pass'''
            pass

        '''for i in range(int(1500/batchsize)+1):

            if i != int(1500/batchsize):
                stant=i*batchsize
                end=(i+1)*batchsize
                picture1[:]=img2[stant:end]
                picture_ans1[:]=train_ans2[stant:end]
                _=sess.run([optimizer],feed_dict={img:picture1,y:picture_ans1,keep_prob1:0.8,keep_prob2:0.3})
                pass
            
            else:
                stant=i*batchsize
                picture3[:]=img2[stant:]
                picture_ans3=train_ans2[stant:]
                _=sess.run([optimizer],feed_dict={img:picture3,y:picture_ans3,keep_prob1:0.8,keep_prob2:0.3})
                pass
            pass'''

        
        for i in range(int(3488/batchsize)):
            
            #if i != int(3488/batchsize):
            stant=i*batchsize
            end=(i+1)*batchsize
            picture1[:]=total_img[stant:end]
            picture_ans1[:]=total_ans[stant:end]
            loss=sess.run([abs_loss],feed_dict={img:picture1,y:picture_ans1,keep_prob1:1,keep_prob2:1})
            all_loss1=all_loss1+loss[0]
            #pass
            
            '''else:
                stant=i*batchsize
                picture2[:]=total_img[stant:]
                picture_ans2[:]=total_ans[stant:]
                loss=sess.run([abs_loss],feed_dict={img:picture2,y:picture_ans2,keep_prob1:1,keep_prob2:1})
                all_loss1=all_loss1+loss[0]
                pass'''
            pass

        '''for i in range(int(1500/batchsize)+1):

            if i != int(1500/batchsize):
                stant=i*batchsize
                end=(i+1)*batchsize
                picture1[:]=img2[stant:end]
                picture_ans1[:]=train_ans2[stant:end]
                loss=sess.run([abs_loss],feed_dict={img:picture1,y:picture_ans1,keep_prob1:1,keep_prob2:1})
                all_loss2=all_loss2+loss[0]
                pass
            
            else:
                stant=i*batchsize
                picture3[:]=img2[stant:]
                picture_ans3=train_ans2[stant:]
                loss=sess.run([abs_loss],feed_dict={img:picture3,y:picture_ans3,keep_prob1:1,keep_prob2:1})
                all_loss2=all_loss2+loss[0]
                pass
            pass'''
        #print(loss[0]*400)
        #print(all_loss1)
        all_loss1=all_loss1/int(3488/batchsize)
        #all_loss2=all_loss2/int(1500/batchsize)+1
        #total_loss=(all_loss1+all_loss2)/2
        #print("all_loss1:"+str(all_loss1*400))
        #print("all_loss2:"+str(all_loss2*400))
        print("train loss : "+str(all_loss1*128))

        testimg1,test_ans1= randomlist(439,testimg,test_ans)
        test_loss=sess.run([abs_loss],feed_dict={img:testimg1,y:test_ans1,keep_prob1:1,keep_prob2:1})
        print("test loss : "+str(test_loss[0]*128))
        end = time.time()
        elapsed = end - start
        print("Time taken: "+str(int(elapsed))+"seconds.")
        if each>9000 and each!=0:
            saver.save(sess,'./model', global_step=each)
            #saver.save(sess, os.path.join(os.getcwd(), 'model.ckpt'))
        #if each%500==0 and each!=0 : #and r1==0:
            #learning_rate_number=learning_rate_number*0.1
        '''if each%400 == 0 and each !=0 and each <=800:
            batchsize=int(batchsize/2)
            picture1 = np.zeros((batchsize,128,128,1))
            picture2 = np.zeros((3488-int(3488/batchsize)*batchsize,128,128,1))
            picture_ans1 = np.zeros((batchsize,238))
            picture_ans2 = np.zeros((3488-int(3488/batchsize)*batchsize,238))'''
            

       

        
        























    



