#!/usr/bin/env python
# -*- coding:utf-8 -*-
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from .utils import context_wrapper
import requests
from requests.auth import HTTPBasicAuth
#import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import tensorflow as tf
import json
import os
import time
from rest_framework_mongoengine import generics
from .serializers import coordinateSerializer
import pymongo
import urllib.parse

class dicom_predict():
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta',
            clear_devices=True)
            saver.restore(self.sess, loc)
            self.img = self.graph.get_operation_by_name('img').outputs[0]
            self.keep_prob1 = self.graph.get_operation_by_name('keep_prob1').outputs[0]
            self.keep_prob2 = self.graph.get_operation_by_name('keep_prob2').outputs[0]
            self.output = tf.get_collection("output")[0]

    def run(self,x):
        self.x=x
        self.tag=self.sess.run(self.output,feed_dict={self.img:self.x,self.keep_prob1:1.0,self.keep_prob2:1.0})
        return self.tag

class dicom_img():
    def __init__(self,dicom_name):
        self.dicom_name=dicom_name
        os.path.isfile(self.dicom_name)
        self.ds= sitk.ReadImage(self.dicom_name)
        self.img_array=sitk.GetArrayFromImage(self.ds)
        self.frame_num,self.width,self.height = self.img_array.shape
        
    def img_take(self):
        self.img_tmp = np.zeros([512,512])
        self.img_tmp = self.img_array[0,:,:]
        self.img_tmp = self.img_tmp.astype(np.float32)
        self.img_tmp = (self.img_tmp-self.img_tmp.min())/(self.img_tmp.max()-self.img_tmp.min())
        self.img_tmp = (self.img_tmp*255).astype(np.uint8)
        self.img_tmp = cv2.resize(self.img_tmp, (128,128))
        self.img_tmp = self.img_tmp.reshape(128,128,1)
        self.img_tmp2= np.zeros((1,128,128,1))
        self.img_tmp2[0]=self.img_tmp
        return self.img_tmp2,self.width,self.height


# Create your views here.
class predict(APIView):
    permission_classes = (AllowAny,)
    def get(self, request, format=None):
        
        ID = self.request.query_params.get('ID',"")
        account = self.request.query_params.get('account',"")
        password = self.request.query_params.get('password',"")
        #studyUID = self.request.query_params.get('studyUID',"")
        #seriesUID = self.request.query_params.get('seriesUID',"")
        #objectUID = self.request.query_params.get('objectUID',"")
        if account == 'cyorthanc' and password == 'cy123orthanc':
            url="https://orthanc.dicom.org.tw/dicom-web/studies/*/series/*/instances/?PatientID="+str(ID)
            r = requests.get(url)
            r=r.json()
            #r=json.loads(r.content)
            studyUID = str(r[0]['0020000D']['Value'][0])
            seriesUID = str(r[0]['0020000E']['Value'][0])
            objectUID = str(r[0]['00080018']['Value'][0])
        
            url="https://orthanc.dicom.org.tw/wado/?requestType=WADO&studyUID="+studyUID+'&seriesUID='+seriesUID+"&objectUID="+objectUID+"&contentType=application%2Fdicom"
            session_requests = requests.session()
            #print(url)
            #print("\\n")
            r = requests.get(url,verify= False,auth=HTTPBasicAuth(account, password))
            #r=str(r,'utf-8')
            #print("\\n")
            #print(chardet.detect(r.content))
            #r.encoding = 'utf-8'
            with open("web.dcm", "w",encoding="utf-8") as code:
                code.write(r.content)
                pass
            #time.sleep(5)
            img_dicom=dicom_img("web.dcm")
            img_tmp,width,height=img_dicom.img_take()

            with tf.Session() as sess:
                one_model=dicom_predict('prediction\onemodel\model-27559')
                predict_tag=one_model.run(img_tmp)
                pass

            x_tag=predict_tag[0][0::2]*128
            y_tag=predict_tag[0][1::2]*128

            x_tag=x_tag/128*width
            y_tag=y_tag/128*height
            
            username = urllib.parse.quote_plus('romongo')
            password = urllib.parse.quote_plus('jp8t6M15aS')
            myclient = pymongo.MongoClient("mongodb://"+username+":"+password+"@mongo:27017/")
            mydb = myclient["test"]
            mycol = mydb["test"]
            myquery = { "PID": str(ID)}
            mydoc = mycol.find_one(myquery)
            if mydoc != None:
                data=[{'x':mydoc['x'],'y':mydoc['y'],'width':width,'height':height}]
                return Response(context_wrapper(200, data=data),status=status.HTTP_200_OK)
            else:
                mydict = { "PID": str(ID), "x": x_tag.tolist(), "y": y_tag.tolist() }
                mycol.insert_one(mydict)
                data={'x':x_tag.tolist(),'y':y_tag.tolist(),'width':width,'height':height}
                #data=json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
                return Response(context_wrapper(200, data=data),status=status.HTTP_200_OK)
        else:
            data = context_wrapper(9991, errors='account not have')
            return Response(data, status.HTTP_400_BAD_REQUEST)

class change(APIView):
    def get(self, request, format=None):
        permission_classes = (AllowAny,)
        ID = self.request.query_params.get('ID',"")
        account = self.request.query_params.get('account',"")
        password = self.request.query_params.get('password',"")
        x = self.request.query_params.get('x',"")
        y = self.request.query_params.get('y',"")
        if account == 'cyorthanc' and password == 'cy123orthanc':
            
            username = urllib.parse.quote_plus('romongo')
            password = urllib.parse.quote_plus('jp8t6M15aS')
            myclient = pymongo.MongoClient("mongodb://"+username+":"+password+"@mongo:27017/")
            mydb = myclient["test"]
            mycol = mydb["test"]
            myquery = { "PID": str(ID) }
            mydict = {"$set": {"x": x.tolist(), "y": y.tolist() }}
            mycol.update_one(myquery, mydict)
            data1={'result':'ok'}
            return Response(context_wrapper(200, data=data1),status=status.HTTP_200_OK)

        else:
            #data={'result':'account not have'}
            data1 = context_wrapper(9991, errors='account not have')
            return Response(data1, status.HTTP_400_BAD_REQUEST)


