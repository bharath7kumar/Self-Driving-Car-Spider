import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import numpy as np 

from PID import pidcontrol
cap=cv2.VideoCapture('output1.avi')


maximum=0
while(cap.isOpened()):
    _,img=cap.read()
#print(img.shape)
    roi_img=img[120:480,0:640]
   # print(roi_img.shape)

    kernel = np.ones((3,3),np.uint8)

    hsv=cv2.cvtColor(roi_img,cv2.COLOR_BGR2HSV)

    lower_black=np.array([0,0,0])
    upper_black=np.array([255,255,60])

    mask=cv2.inRange(hsv,lower_black,upper_black)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    erosion=cv2.erode(dilation,kernel,iterations=2)
    dilation2 = cv2.dilate(erosion,kernel,iterations = 4)
    #cv2.imshow('mask',dilation2)

    #res=cv2.bitwise_and(img,img,mask=dilation2)

    #cv2.imshow('original_roi',roi_img)
    #cv2.imshow('res',res)

    kernel2=np.ones((11,11),np.uint8)
    lower_pink=np.array([80,80,80])
    upper_pink=np.array([255,255,255])
    mask2=cv2.inRange(hsv,lower_pink,upper_pink)
    dilation3=cv2.dilate(mask2,kernel,iterations=10)
    #cv2.imshow('pink mask',dilation3)

    final_res=cv2.bitwise_or(dilation3,dilation2)
    #cv2.imshow('final_res',final_res)

    final_res_inv=cv2.bitwise_not(final_res)
    cv2.imshow('final_res_inv',final_res_inv)
    
    pid=pidcontrol(final_res_inv) 
    
    #pid.speed() 
    pid.contourarea(roi_img)
    #=pid.nopixels()
    
    #temp=error
    #if(max(maximum,abs(temp))!=maximum):
    	#maximum=temp

    #print(maximum)

    if cv2.waitKey(10) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#print("Maximum error:",maximum)
