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



class CBISImagePreprocessor(object):

    def __init__(self,mlo_muscle):
        self.mlo_muscle = mlo_muscle

    def convert_dicom_png(self,dicom_path,dicom_folder_path):

        for n, image in enumerate(dicom_path):

            ds = dicom.dcmread(os.path.join(dicom_folder_path,image))  # forces the decompression
            pixel_array_numpy = ds.pixel_array
            if(PNG == False):
                image = image.replace('.dcm','.jpg')
            else:
                image = image.replace('.dcm','.png')
            cv2.imwrite(os.path.join(dicom_folder_path,image),pixel_array_numpy)

            if(n % 50 == 0):
               print('{} image converted'.format(n))


    '''
        Caso ocorra overflow na primeira função utilizar essa
    '''
    def dicom_to_png(self,inputdir):

        for f in inputdir:
            ds = dicom.read_file(f) 
            img = ds.pixel_array 
            cv2.imwrite(f.replace('.dcm','.png'),img)

    def read_dicom_image(self, image_path):

    	ds = dicom.dcmread(image_path)
    	plt.imshow(ds.pixel_array)
    	plt.show()
     


    def remove_files_img(self,path):

        dir_ = glob.glob('path\\*.png')
        for file in dir_ :
            if file.endswith(""):
                os.remove(file)
              
    			
    def rename_files_img(self,path):
    	for i,filename in enumerate(os.listdir(path)):
                os.rename(os.path.join(path,filename),os.path.join(path,filename.replace('2.png', '.png')))


    def extract_files_f(self,src_path,out_path):
 
        for dirpath, dirnames, filenames in os.walk(src_path):
            for i,filename in enumerate (filenames):
                if filename.endswith(".dcm"):
                    src = os.path.join(dirpath , filename)
                    dirw =dirpath[:71]
                    new_ = dirw[40:]
                    dest = os.path.join(out_path,str(new_) + filename)
                    shutil.copy2(src, dest)

    
    def select_largest_obj(self, img_bin, lab_val=255, fill_holes=False, 
                           smooth_boundary=False, kernel_size=15):

        '''
        Função retirada de: https://github.com/yuyuyu123456/CBIS-DDSM
        Selecione o maior objeto de uma imagem binária e opcionalmente
        preenche 'buracos' na imagem e suaviza seu limite.
        Args:
            img_bin (2D array): 2D numpy array da imagem binária.
            lab_val ([int]): integer value -> label (largest 
                    object). Default = 255.
            fill_holes ([boolean]): se preenche os buracos dentro do maior 
                    objeto ou não. Default = false.
            smooth_boundary ([boolean]): se suaviza o limite do 
                    maior objeto usando abertura morfológica ou não.
                    Default = false.
            kernel_size ([int]): o tamanho do kernel usado para operação morfológica.
             Default = 15.
        Returns:
        	Binary mask (:

        '''
        n_labels, img_labeled, lab_stats, _ = \
            cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                             ltype=cv2.CV_32S)
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val

        if fill_holes:
            bkg_locs = np.where(img_labeled == 0)
            bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                          newVal=lab_val)
            holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
            largest_mask = largest_mask + holes_mask
        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
                                            kernel_)
            
        return largest_mask

    @staticmethod
    def max_pix_val(dtype):

        if dtype == np.dtype('uint8'):
            maxval = 2**8 - 1
        elif dtype == np.dtype('uint16'):
            maxval = 2**16 - 1
        else:
            raise Exception('Unknown dtype found in input image array')
        return maxval

    def suppress_artifacts(self, img, global_threshold=.05, fill_holes=False, 
                           smooth_boundary=True, kernel_size=15):
        '''
          Função retirada de: https://github.com/yuyuyu123456/CBIS-DDSM
        	Mascarar artefatos de uma imagem de entrada
        	(objetos indesejáveis, ticket etc.)
        Args:
            img (matriz 2D): imagem de entrada como numpy array.
            global_threshold ([int]):  Default = 18.
            kernel_size ([int]): tamanho do kernel para operações morfológicas. 
                    Default = 15.
        Retorna:
            uma tupla de (output_image, breast_mask). 
            Ambos são matrizes numpy 2D (:

        '''
        maxval = self.max_pix_val(img.dtype)
        if global_threshold < 1.:
            low_th = int(img.max()*global_threshold)
        else:
            low_th = int(global_threshold)
        _, img_bin = cv2.threshold(img, low_th, maxval=maxval, 
                                   type=cv2.THRESH_BINARY)
        breast_mask = self.select_largest_obj(img_bin, lab_val=maxval, 
                                              fill_holes=True, 
                                              smooth_boundary=True, 
                                              kernel_size=kernel_size)
        img_suppr = cv2.bitwise_and(img, breast_mask)

        return (breast_mask)

       
    def binarization_muscle_supressartifacts(self,mlo_muscle):

        for j,i in enumerate (mlo_muscle):

            mammo_org = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
            mammo_med_blurred = cv2.medianBlur(mammo_org,3)
            res = hstack((mammo_org, mammo_med_blurred))
            global_threshold = 18
            _,mammo_binary = cv2.threshold(mammo_org,global_threshold,maxval = 255,type = cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
            mammo_breast_mask = self.select_largest_obj(mammo_binary,lab_val = 255,
                          fill_holes=False,
                          smooth_boundary=False, kernel_size=5)
            mammo_arti_supr = cv2.bitwise_and(mammo_med_blurred,mammo_breast_mask)
            cv2.imwrite(i,mammo_arti_supr)
    
    def extract_pectoral_muscle(self):

        #Based on: https://github.com/yuyuyu123456/CBIS-DDSM
        for j,i in enumerate (mlo_muscle):

            mammo_org = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
            mammo_med_blurred = cv2.medianBlur(mammo_org, 3)
            #res = hstack((mammo_org, mammo_med_blurred))
            global_threshold = 18
            _,mammo_binary = cv2.threshold(mammo_med_blurred,global_threshold,maxval = 255,type = cv2.THRESH_BINARY)
            mammo_breast_mask = self.select_largest_obj(mammo_binary,lab_val = 255,
                          fill_holes=False,
                          smooth_boundary=False, kernel_size=15)
            mammo_arti_supr = cv2.bitwise_and(mammo_med_blurred,mammo_breast_mask)
            mammo_breast_equ = cv2.equalizeHist(mammo_arti_supr)
            pect_high_iten_thres = 200
            _, pect_binary_thres = cv2.threshold(mammo_breast_equ, pect_high_iten_thres,
                                    maxval=255,type=cv2.THRESH_BINARY)
            largest_mask = self.select_largest_obj(pect_binary_thres,255,False,False,15)
            pect_marker_img = np.zeros(largest_mask.shape,dtype=np.int32)
            pect_mask_init = self.select_largest_obj(largest_mask,lab_val=255,
                                            fill_holes=False, smooth_boundary=False)
            kernel_ = np.ones((3,3),dtype=np.uint8)
            n_erosions = 7
            pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_erosions)
            pect_marker_img[pect_mask_eroded > 0] = 255
            n_dilations = 7
            pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_dilations)
            pect_marker_img[pect_mask_eroded  == 0] = 128
            pect_marker_img[mammo_breast_mask == 0] = 64
            mammo_breast_equ_3c = cv2.cvtColor(mammo_breast_equ, cv2.COLOR_GRAY2BGR)
            cv2.watershed(mammo_breast_equ_3c, pect_marker_img)
            pect_mask_watershed = pect_marker_img.copy()
            mammo_breast_equ_3c[pect_mask_watershed == -1] = (0, 0, 255)
            pect_mask_watershed[pect_mask_watershed == -1] = 0
            breast_only_mask = pect_mask_watershed.astype(np.uint8)
            breast_only_mask[breast_only_mask != 128] = 0
            breast_only_mask[breast_only_mask == 128] = 255
            kn_size = 25  # <<= para to tune!
            kernel_ = np.ones((kn_size, kn_size), dtype=np.uint8)
            breast_only_mask_smo = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
            mammo_breast_only = cv2.bitwise_and(mammo_breast_equ, breast_only_mask_smo)
            cv2.imwrite(i,mammo_breast_only)

                

