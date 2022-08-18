import cv2
import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
import glob
import pydicom as dicom
from os import rename
from pydicom.encaps import encapsulate
from PIL import Image, ImageFilter, ImageOps
import os, sys
import shutil
import unicodedata
import argparse
import pandas as pd

class DetectionPoints(object):

    def __init__(self):
        pass
    
    def pointII_detection(self,im,width):
        
        for i in range(0,width):
            if ((im[20, i]).any() == 0):
                po = i
                P2 = [po,0]
                po = int(po)
                return P2

    def point_po(self,im,width):
          
        for i in range(0,width):
            if ((im[20, i]).any() == 0):
                po = i
                P2 = [po,0]
                po = int(po)
                return po
    
    def pointIV_detection(self,im,width,temp):
        for i in range(0,width):
            if((im[20, i]).any() == 0):
                po = i
                P4 = [po,temp]
                temp =int(temp)
                return P4
    
    def temp_point(self,im,width,temp):

        for i in range(0,width):
            if((im[20, i]).any() == 0):
                po = i
                P4 = [po,temp]
                temp =int(temp)
                return temp

    def channels_height_width(self,img):
    
        height = img.shape[0]
        width = img.shape[1]
        temp = (height * 0.60)
        return height,width,temp


    def cropped_img_function(self):

        for image in mlo_muscle:
            im  = cv2.imread(image,0)
            height,width,temp = self.channels_height_width(im)
            P2 = self.pointII_detection(im,width)
            P4= self.pointIV_detection(im,width,temp)
            po = self.point_po(im,width)
            aux = self.temp_point(im,width,temp)
            #print(po,aux)
            cropped_image = im[0:aux, 0:po]
            cv2.imwrite(image,cropped_image)

    
    def resize_images_out(self):
        
        path = glob.glob('/dir/*.png')

        for x in path:    
            gray_ = cv2.imread(x)
            mask = x.replace("Benign Mass","Benign Mask")
            bin_ = cv2.imread(mask)
            width = gray_.shape[1]
            height = gray_.shape[0]
            print(width,height)
            dim = (width,height)
            resize = cv2.resize(bin_,dim,interpolation = cv2.INTER_AREA)
            cv2.imwrite(x,resize)

    def boudingbox_roi_image(self,path):

        for i, im in enumerate(path):
            image = cv2.imread(im,0)
            contours,hierarchy = cv2.findContours(image, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                count +=1
                x,y,w,h = cv2.boundingRect(cnt)
                roi = image[y:y+h,x:x+w]
                cv2.imwrite(im, roi)

    
    def bitwise_mask_and_gray(self):

        path = glob.glob('/dir/malignant1/*.png')

        for i in path:
            gray_ = cv2.imread(i,0)
            bin_ = i.replace("malignant1","malignant2")
            mask_ = cv2.imread(bin_,0)
            op_and = cv2.bitwise_and(gray_,mask_)
            cv2.imwrite(i,op_and)
            



    