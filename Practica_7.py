#Edwing Alexis Casillas Valencia.   19110113.   7E1.    Práctica 7 visión artificial
#Meter las funciones al main
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

def smooth():
    #cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)

        kernel=np.ones((15,15),np.float32)/255
        filt=cv2.filter2D(res,-1,kernel)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def blur():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)

        filt=cv2.GaussianBlur(res,(15,15),0)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def f_Median():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)

        filt=cv2.medianBlur(res,15)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def bilateral():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)

        filt=cv2.bilateralFilter(res,15,75,75)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def f_Erosion():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)
        kernel=np.ones((5,5),np.uint8)

        filt=cv2.erode(mask_y,kernel,iterations=1)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def f_Dilatacion():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)
        kernel=np.ones((5,5),np.uint8)

        filt=cv2.dilate(mask_y,kernel,iterations=1)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def f_Opening():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)
        kernel=np.ones((5,5),np.uint8)

        filt=cv2.morphologyEx(mask_y,cv2.MORPH_OPEN,kernel)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def f_Closing():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)
        kernel=np.ones((5,5),np.uint8)

        filt=cv2.morphologyEx(mask_y,cv2.MORPH_CLOSE,kernel)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def morfo():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)
        kernel=np.ones((5,5),np.uint8)

        filt_o=cv2.morphologyEx(mask_y,cv2.MORPH_CLOSE,kernel)
        filt=cv2.morphologyEx(filt_o,cv2.MORPH_OPEN,kernel)
        #filt=filt_o+filt_p

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

def lin():
    while(1):
        _, frame = cap.read()
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        #hue saturation value
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([60,255,255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res = cv2.bitwise_and(frame,frame, mask= mask_y)

        kernel=np.ones((15,15),np.float32)/255
        filt_1=cv2.filter2D(res,-1,kernel)
        filt_2=cv2.GaussianBlur(res,(15,15),0)
        filt=cv2.medianBlur(filt_2,15)

        cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask_y)
        cv2.imshow('res',res)
        cv2.imshow('redux',filt)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    cv2.destroyAllWindows()
    cap.release()

print('Selecciona la reducción de ruido: \n')
redo=int(input(' 1) Lineal\n 2) Morfológico\n'))

if redo==1:
    print('Selecciona el metodo para la reducción de ruido:\n')
    met=int(input('1) Smoothed\n2) Blur\n3) Median\n 4) Bilateral\n'))
    if met==1:
        smooth()
    elif met==2:
        blur()
    elif met==3:
        f_Median()
    elif met==4:
        bilateral()
    else:
        lin()
else:
    print('Selecciona el metodo para la reducción de ruido:\n')
    met=int(input('1) Erosion\n2) Dilatation\n3) Opening\n 4) Closing\n'))
    if met==1:
        f_Erosion()
    elif met==2:
        f_Dilatacion()
    elif met==3:
        f_Opening()
    elif met==4:
        f_Closing()
    else:
        morfo()
#smooth()
