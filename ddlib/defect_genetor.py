import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import cv2 as cv
import numpy as np
import os
import torch
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import  cv2
from  ddlib.utils import * # 不会导入私有成员
from  ddlib.utils import _pair
from collections import Iterable
from ddlib.transforms import *


class DefectGeneratorBase(object):

    def gen_defect(self,shape):
        """产生随机分布，填充缺陷区域"""
        mean = random.randint(-125, 125)
        fluct = random.randint(1, 100)
        low = mean - fluct  # +(mean-fluct<0)*abs(mean-fluct)
        high = mean + fluct  # -(mean+fluct>255)*abs(255-(mean+fluct))
        defect = np.random.randint(low, high, shape)
        defect = defect.astype(np.uint8)
        return defect

    def rand_move_roi(self,roi,image):
        h,w = get_img_shape(image)
        Rows, Cols = np.nonzero(roi)
        w_roi = np.max(Cols) - np.min(Cols)
        h_roi = np.max(Rows) - np.min(Rows)
        assert h>h_roi and w>w_roi,print(h,w ,"  " , h_roi,w_roi)
        offset_row=random.randint(0, h - h_roi - 1)
        offset_col = random.randint(0, w - w_roi - 1)
        Rows,Cols=Rows+offset_row- np.min(Rows),Cols+offset_col- np.min(Cols)
        return Rows,Cols

    def load_img(self, img_path):
        img = Image.open(img_path).convert('L')
        return np.array(img,dtype=np.uint8)

    def defect_fit_img(self,defect, img_shape):
        defect_shape = get_img_shape(defect)
        assert defect_shape[0] < img_shape[0] and defect_shape[1] < img_shape[1]
        pad_up = np.random.randint(0, img_shape[0] - defect_shape[0] + 1)
        pad_down = img_shape[0] - defect_shape[0] - pad_up
        pad_left = np.random.randint(0, img_shape[1] - defect_shape[1] + 1)
        pad_right = img_shape[1] - defect_shape[1] - pad_left
        defect = np.pad(defect, ((pad_up, pad_down), (pad_left, pad_right)), 'constant', constant_values=0)
        return defect

    def crop(self,mask):
        Rows, Cols = np.nonzero(mask)
        mask = np.array(mask)[np.min(Rows):np.max(Rows) + 1, np.min(Cols):np.max(Cols) + 1]
        return  mask

    def to_bin(self,mask,thred=0):
        mask=np.where(mask>thred,1,0).astype(np.uint8)
        assert np.max(mask)>0
        return mask
    def crop_roi(self,img,mask):
        """
        :param img:
        :param mask:
        :return:
        """
        Rows, Cols = np.nonzero(mask)
        img=np.array(img)[np.min(Rows):np.max(Rows)+1,np.min(Cols):np.max(Cols)+1]
        mask = np.array(mask)[np.min(Rows):np.max(Rows) + 1, np.min(Cols):np.max(Cols) + 1]
        return img,mask


    def load_with_shape(self,img_path,shape):
        """
        如果图片尺寸小于shape，就上采样，并保持宽高atio布边
        :param img_path:
        :param shape:
        :return:
        """
        img=np.array(self.load_img(img_path),dtype=np.uint8)
        if np.array(img).shape[0] < shape[0] or np.array(img).shape[1] < shape[1]:
            rate=max(shape[0]/img.shape[0],shape[1]/img.shape[1])
            img=Image.fromarray(img).resize((int(img.shape[1]*rate),int(img.shape[0]*rate)),resample=Image.BILINEAR)
        return np.array(img)

    def rand_paint_defect_on_img(self,defect,roi,img):
        Rows_defect, Cols_defect = np.nonzero(roi)
        Rows_obj, Cols_obj = self.rand_move_roi(roi, img)
        defect_img = img.copy().astype(np.float)
        defect = np.array(defect)
        if random.random() > 0.7:
            defect_img[Rows_obj, Cols_obj] = defect[Rows_defect, Cols_defect]
        else:
            defect_img[Rows_obj, Cols_obj] = defect[Rows_defect, Cols_defect] + defect_img[Rows_obj, Cols_obj]
        defect_img = np.clip(defect_img, 0, 255).astype(np.uint8)
        defect_roi = np.zeros_like(img)
        defect_roi[Rows_obj, Cols_obj] = 1
        return defect_img,defect_roi

class DefectGenerator(DefectGeneratorBase):
    def __init__(self,defect_dir=None,noise_dir=None,roi_dir=None,**kwargs):

        if defect_dir ==None:
            defect_dir = os.path.join(get_cur_path(), "imgs/defect")
        if noise_dir == None:
            noise_dir = os.path.join(get_cur_path(), "imgs/noise")
        if roi_dir==None:
            roi_dir = os.path.join(get_cur_path(), "imgs/mask")


        self.init_transforms_defect()
        self.defect_paths=list_folder(defect_dir,True)
        self.noise_paths=list_folder(noise_dir)
        self.roi_paths = list_folder(roi_dir)

    def __call__(self, img):
        img=np.array(img)
        rand=random.random()
        if rand<0.6:
            defect=self.load_img(random.choice(self.defect_paths))
            roi=self.to_bin(defect)
        elif 0.6<=rand<0.8:
            roi = self.to_bin(self.load_img(random.choice(self.roi_paths)))
            roi=self.crop(roi)

            noise = self.load_with_shape(random.choice(self.noise_paths),shape=roi.shape)
            Rows_defect, Cols_defect = self.rand_move_roi(roi, noise)
            defect=noise[np.min(Rows_defect):np.max(Rows_defect)+1, np.min(Cols_defect):np.max(Cols_defect)+1]

        else:
            roi = self.to_bin(self.load_img(random.choice(self.roi_paths)))
            roi = self.crop(roi)
            defect = self.gen_defect(shape=roi.shape)

        #对缺陷进行各种变换
        for trans_func in self.trans:
            defect,roi=trans_func(defect,roi)
        #最小外接矩形剪切
        assert np.max(roi) > 0
        defect,roi=self.crop_roi(defect,roi)

        #缺陷写入
        defect_img,defect_roi=self.rand_paint_defect_on_img(defect,roi,img)
        return defect_img,defect_roi

    def init_transforms_defect(self):
        self.trans=[
            RandomRotate(probability=1),
            RandomGrayLevelTrans(mean_factor=None,mean=(5,30),probability=1),
            RandResize(scale_fator=None,scale=(4,20),ratio=(0.2,5),probability=1)
        ]



from    ddlib import  get_cur_path
if __name__ == '__main__':

    img_dir="E:\ZTB\复赛数据\data-part1\data\TC_Images\TC_Images\part1\TC_Images"
    defectGenerator=DefectGenerator()

    imgs=list_folder(img_dir)
    for img1 in imgs:
        img=Image.open(img1).convert('L')
        defect_img,mask=defectGenerator(img)

        # plt_show_imgs([img,mask,defect_img])
