import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk


ds = sitk.ReadImage("2.dcm")
img_array = sitk.GetArrayFromImage(ds)
print(img_array.shape)
frame_num, width, height = img_array.shape
#i=200
#j=1800
#print(img_array)
print(frame_num)
print(width)
print(height)
print("=======================================================")

img_tmp = np.zeros([512,512])
img_tmp = img_array[0,:,:]
img_tmp = img_tmp.astype(np.float32)

print(img_tmp)
print(img_tmp.min())
print(img_tmp.max())
print("=======================================================")
#a=img_tmp>j+500
#print(a[0])
#img_tmp[a]=0
#b=img_tmp<j-500
#img_tmp[b]=0
#img_tmp[img_tmp<300]=0
#print(img_tmp)
#img_tmp= (img_tmp-i)
#img_tmp[img_tmp<0]=0
img_tmp = (img_tmp-img_tmp.min())/(img_tmp.max()-img_tmp.min())

#img_tmp = (img_tmp-img_tmp.min())/(img_tmp.max()-img_tmp.min())
#print(img_tmp)
#print(img_tmp[0][0])
img_tmp = (img_tmp*255).astype(np.uint8)
#print(img_tmp)
img_tmp=cv2.resize(img_tmp, (400,400))
cv2.imwrite('1.jpg',img_tmp)
cv2.imshow('img_mha',img_tmp)  #圖片顯示
cv2.waitKey(0)  #等待用戶輸入
cv2.destroyAllWindows()  #註銷顯示的窗口


