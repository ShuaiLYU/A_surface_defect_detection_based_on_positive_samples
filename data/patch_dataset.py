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
from demo_.ztb_defect import defect_generator,fun
class ROI_transforms(object):
    def __init__(self, size=(128, 128)):
        super(ROI_transforms).__init__()
        self.size = size
        self.area = self.size[0] * self.size[1]
        self.prosepect_pmax = 0.02
        self.morph_method = [cv.MORPH_DILATE, cv.MORPH_ERODE, cv.MORPH_OPEN, cv.MORPH_CLOSE]
        self.gray_mask_method = [self.rand_mask, self.gauss_mask, self.const_mask]

    def transform_bit(self, x):
        x = self.get_shape_mask(x)
        x = self.rotate(x, angle=[0, 359])

        if random.random() > 0.5:
            x = self.scale(x, [0.5, 2], [0.5, 2])
        if random.random() > 0.5:
            x = self.random_morphological_processing(x, k_size=[3, 11])
        # x = self.align_img(x)
        x = self.crop_or_pad(x)
        _, x = cv.threshold(x, 20, 255, cv.THRESH_BINARY)
        return x

    def transform_gray(self, x):
        x = self.get_shape_mask(x)
        x = self.rotate(x, angle=[0, 359])
        if random.random() > 0.5:
            x = self.scale(x, [0.5, 2], [0.5, 2])
        # x = self.align_img(x)
        x = self.crop_or_pad(x)
        return x


    def get_new_roi(self, mask):
        """

        :param mask: (ndarray.uint8)[Height, Width]
        :return:
        """
        # 增加灰度图和二值图的判断
        # 存在20-230灰度值像素则认定为灰度图
        _, mask_dist = cv.threshold(mask, 20, 255, cv.THRESH_TOZERO)
        _, mask_dist = cv.threshold(mask_dist, 230, 255, cv.THRESH_TOZERO_INV)
        if np.count_nonzero(mask_dist) < 5:
            # 二值图处理
            # 1.mask增强
            mask = self.transform_bit(mask)
            # 2.灰度赋值
            mask = mask.astype(np.float32) / 255
            low = np.random.randint(0, 150)
            high = low + np.random.randint(50, 105)
            mask = self.gray_mask_method[np.random.randint(0, len(self.gray_mask_method) - 1)](mask, low, high)
            if np.random.random() < 0.7:
                # 平滑随机噪声
                k = np.random.randint(1, 5) * 2 + 1
                cv.GaussianBlur(mask, (k, k), k, mask)
        else:
            # 灰度图处理
            # 增强
            mask = self.transform_gray(mask)
            scale = 0.8 + 0.4 * np.random.rand()
            offset = np.random.randint(-10, 10)
            # 随机线性变换
            cv.convertScaleAbs(mask, mask, scale, offset)
        mask = mask.astype(np.uint8)
        # if mask.shape != self.size:
        #     cv.imshow("1", mask)
        #     cv.imshow("2", self.crop_or_pad(mask))
        #     cv.waitKey(0)

        return mask


    def get_shape_mask(self, x):
        if np.count_nonzero(x) < 20:
            return np.ones((np.random.randint(5, 15), np.random.randint(5, 15)), dtype=np.uint8) * 255
        Row = np.argwhere(np.sum(x, axis=0) != 0)
        Col = np.argwhere(np.sum(x, axis=1) != 0)
        x = x[np.min(Col): np.max(Col) + 1, np.min(Row): np.max(Row) + 1]
        # 控制像素数量
        while np.count_nonzero(x) > self.area * self.prosepect_pmax:
            scale = np.random.random()
            scale = scale if scale > 0.5 else 0.5
            x = cv.resize(src=x, dsize=(int(x.shape[1]*scale), int(x.shape[0]*scale)), interpolation=cv.INTER_NEAREST)
        return x

    # 旋转

    def rotate(self, x, angle=0):
        H, W = x.shape
        if isinstance(angle, list):
            assert len(angle) == 2
            angle = np.random.randint(angle[0], angle[1])

        x = np.pad(x, ((W//2, W//2), (H//2, H//2)), mode="constant", constant_values=0)
        H, W = x.shape
        m = cv.getRotationMatrix2D((W//2, H//2), angle, scale=1)
        x = cv.warpAffine(x, m, (x.shape[1], x.shape[0]))
        x = self.get_shape_mask(x)
        return x

    # 形态学处理
    def random_morphological_processing(self, x, k_size=3):
        if isinstance(k_size, list):
            k_size = np.random.randint(k_size[0], k_size[1])
        k_size = k_size // 2 * 2 + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        param = {"src": x, "kernel": element}
        param["op"] = self.morph_method[random.randint(0, len(self.morph_method) - 1)]
        y = cv.morphologyEx(**param)
        if np.sum(y)//255 < 10:
            return x
        return y

    # 放缩
    def scale(self, x, scaleX_factor=1, scaleY_factor=1):
        if isinstance(scaleX_factor, list):
            assert len(scaleX_factor) == 2
            scaleX_factor = scaleX_factor[0] + (scaleX_factor[1] - scaleX_factor[0]) * np.random.rand()
        if isinstance(scaleY_factor, list):
            assert len(scaleY_factor) == 2
            scaleY_factor = scaleY_factor[0] + (scaleY_factor[1] - scaleY_factor[0]) * np.random.rand()
        cv.resize(x, (int(x.shape[1] * scaleX_factor), int(x.shape[0] * scaleY_factor)), x,
                  interpolation=cv.INTER_LINEAR)
        return x

    # 回归尺寸
    # def align_img(self, x):
    #     # if np.random.random() < 0.2:
    #     #     x = self.resize(x)
    #     # else:
    #     #     x = self.crop_or_pad(x)
    #     #
    #     x = self.crop_or_pad(x)
    #     cv.threshold(x, 20, 255, cv.THRESH_BINARY, x)
    #     return x

    def resize(self, x):
        x = np.resize(x, self.size)
        return x

    def crop_or_pad(self, x):
        y = None
        cnt = 0
        while y is None or np.sum(y)//255 < 10:
            H = x.shape[0] - self.size[0]
            W = x.shape[1] - self.size[1]
            if H < 0:
                H = -H
                pad_top = random.randint(0, H)
                y = np.pad(x, ((pad_top, H - pad_top), (0, 0)), mode="constant", constant_values=0)
            else:
                crop_top = random.randint(0, H)
                y = x[crop_top: crop_top + self.size[0]]
            if W < 0:
                W = -W
                pad_left = random.randint(0, W)
                y = np.pad(y, ((0, 0), (pad_left, W - pad_left)), mode="constant", constant_values=0)
            else:
                crop_left = random.randint(0, W)
                y = y[:, crop_left: crop_left + self.size[1]]
            # crop有时只裁剪到黑色区域,此时直接resize
            if np.sum(y)//255 < 10:
                cnt += 1
                if cnt >= 5:
                    return np.resize(x, self.size).astype(np.uint8)
        return y

    # 随机mask灰度值
    def rand_mask(self, mask, low, high):
        gray_mask = np.random.randint(low, high, mask.shape) * mask
        return gray_mask

    def gauss_mask(self, mask, low, high):
        mask = self.get_shape_mask(mask)
        gauss_x = cv.getGaussianKernel(mask.shape[1], mask.shape[1])
        gauss_y = cv.getGaussianKernel(mask.shape[0], mask.shape[0])
        kyx = np.multiply(gauss_y, np.transpose(gauss_x))

        mask = mask * kyx
        Max = np.max(mask)
        Min = np.min(np.where(mask == 0, Max, mask))


        gray_mask = low + (mask - Min) / (Max - Min) * (high - low)
        gray_mask = np.where(gray_mask > 0, gray_mask, 0)
        gray_mask = self.crop_or_pad(gray_mask)
        return gray_mask

    def const_mask(self, mask, *args):
        gray_mask = mask * np.random.randint(0, 255)
        return gray_mask


# def genRandImg1(size,mask):
#     path = './gg'
#     picture_rand = os.listdir(path)
#     len_rand_picture = len(picture_rand)
#     x = random.randint(0, len_rand_picture - 1)
#     name_image = picture_rand[x]
#     picture = cv.imread(path + '/' + name_image, 0)
#     # print(type(picture))
#     picture = cv.resize(picture, (128,128))
#     # print(picture)
#     # _, mask_pict = cv.threshold(picture, 150, 255, cv.THRESH_BINARY)
#     #
#     # cv2.imshow('image',mask_pict)
#     # cv2.waitKey()
#     picture = picture.astype(np.float)
#     return picture


def get_new_image(img, gray_mask):
    gray_mask = gray_mask.astype(np.float32)
    mask = np.where(gray_mask > 0, 1, 0)
    # mask = np.where(gray_mask > 0, 255, 0)
    # mask1=mask.astype(np.uint8)
    # cv.imshow('mask',mask1)
    # cv.waitKey(5000)
    # cover
    if random.random() > 0.8:

        new_img = (img * (1 - mask) + gray_mask * mask)
    else:

        # new_img = (img * (1 - mask)) + gray_mask * mask * (255 - np.mean(img)) / 255
        new_img = (img * (1 - mask)) + mask * img * (1 + (gray_mask - 127.5) / 127.5)
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img

def smooth_edge(new_img, mask):
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, element)
    mask_erode = cv.morphologyEx(mask, cv.MORPH_ERODE, element)
    mask_edge = ((mask_dilate - mask_erode) / 255).astype(np.float32)
    new_img_gauss = cv.GaussianBlur(new_img, (5, 5), 5)
    return (new_img * (1 - mask_edge) + new_img_gauss * mask_edge).astype(np.uint8)


class DefectiveGenerator(object):
    def __init__(self,dir_Database,shape_Img,Limit_ROI=(20,10000),withDatabase=True):
        """

        :param dir_Database:  缺陷ROI路径
        :param shape_Img:   图片大小[height,width]
        :param Limit_ROI:   ROI外接矩形大小[lower ,upper]
        :param withDatabase:   true:从硬盘读入ROI  false：算法生成ROI
        """
        self.dir_Database=dir_Database
        self.height_Img = shape_Img[0]
        self.width_Img=shape_Img[1]
        self.lowerLimit_ROI=Limit_ROI[0]
        self.upperLimit_ROI=Limit_ROI[1]
        #
        self.roi_transform = ROI_transforms()
        #从数据库读入ROI
        self.names_ROIs,self.num_ROIs=self.loadROIs(self.dir_Database)
        if self.num_ROIs<1:
            print("the dataset is empty!")


    def loadROIs(self,dir):
        # ROIs=os.listdir(dir)
        # 递归遍历文件
        ROIs = list()
        for root, dirs, files in os.walk(dir):
            # print(root)
            for file in files:
                if file.endswith(".bmp") or file.endswith(".PNG"):
                    ROIs.append(os.path.join(root, file))

        num_ROI=len(ROIs)
        print('采用本地ROI个数为{}'.format(num_ROI))
        return ROIs,num_ROI

    def genRandImg(self,size):
        mean=random.randint(-125,125)
        fluct=random.randint(1,100)
        low=mean-fluct  #+(mean-fluct<0)*abs(mean-fluct)
        high=mean+fluct   #-(mean+fluct>255)*abs(255-(mean+fluct))
        img=np.random.randint(low,high,size)
        img=img.astype(np.float)
        return img

    def genRandImg1(self,size):
        path = './gg'
        picture_rand = os.listdir(path)
        len_rand_picture = len(picture_rand)
        x = random.randint(0, len_rand_picture - 1)
        name_image = picture_rand[x]
        picture = cv.imread(path + '/' + name_image, 0)
        # print(type(picture))
        picture = cv.resize(picture, (128,128))
        # cv.imshow('image',picture)
        # cv.waitKey()
        # print(picture)
        # _, mask_pict = cv.threshold(picture, 150, 255, cv.THRESH_BINARY)
        #
        # cv.imshow('image',picture)
        # cv.waitKey(5000)
        # picture = picture.astype(np.float)
        # cv.imshow('image',picture)
        # cv.waitKey(5000)
        return picture

    def apply(self,img,both=False):
        ROI=self.randReadROI()
        # 灰度mask处理
        # 1.最小矩形提取
        # 2.随机旋转和放缩
        # 3.尺寸回归

        # 二值mask处理
        # roi增强
        # 1.最小矩形提取形状
        # 2.随机旋转和放缩
        # 3.形态学处理
        # 4.回归尺寸
        # 返回二值掩模图

        # 返回灰度roi
        ROI_new = self.roi_transform.get_new_roi(ROI)
        ROI_new = np.where(ROI_new > 0, 1, 0).astype(np.uint8)
        #
        # cv.imshow('mask',ROI_new)
        # cv.waitKey(5000)
        if both:
            randd = random.randint(0, 1)
            if randd == 0:
                img_rand = self.genRandImg([self.height_Img, self.width_Img])
            else:
                img_rand = self.genRandImg1([self.height_Img, self.width_Img])
        else:
            img_rand = self.genRandImg1([self.height_Img, self.width_Img])
        img_new = img.astype(np.float).astype(np.uint8)
        rand = random.randint(0, 1)
        # print(img_new.shape)
        # cv.imshow('mask', img_rand)
        if rand == 0:
            # pass
            # print(1-ROI_new)
            img_new = img_new * (1 - ROI_new) + (img_rand * ROI_new)*0.45
        else:
            # img_new = img_new + img_rand * ROI_new
            img_new = img_new * (1 - ROI_new) + (img_rand * ROI_new)*0.45
            # pass

        #  img_new = img_new * (1 - ROI_new) + img_rand * ROI_new
        img_new = np.clip(img_new, 0, 255).astype(np.uint8)
        # ROI_new = (ROI_new * 255).astype(np.uint8)
        # print(img_new.shape)
        # cv.imshow('mask',img_new)
        # cv.imshow('mask', img_rand)
        # cv.waitKey()
        # print(img_new.shape)
        return img_new, ROI_new

        # img_new = get_new_image(img, ROI_new)
        #
        # img_new = smooth_edge(img_new, ROI_new)
        # cv.imshow("img", img)
        # cv.imshow("ROI", ROI)
        # cv.imshow("ROI_new", ROI_new)
        # cv.imshow("img_new", img_new)
        # cv.waitKey(0)


        #
        # img_rand=self.genRandImg([self.height_Img, self.width_Img])
        # img_new=img.astype(np.float)


        # rand = np.random.randint(0, 1)

        # if rand==0:
        #    img_new=img_new*(1-ROI_new)+img_rand*ROI_new
        # else:
        #     img_new = img_new + img_rand * ROI_new
      #  img_new = img_new * (1 - ROI_new) + img_rand * ROI_new
      #   img_new=np.clip(img_new, 0, 255).astype(np.uint8)
      #   ROI_new=(ROI_new*255).astype(np.uint8)
      #   return img_new, ROI_new

    def randReadROI(self):

        while(1):
            rand = random.randint(0, self.num_ROIs - 1)
            name_Img = self.names_ROIs[rand]
            img_Label = cv.imread(name_Img, 0)
            cv.threshold(img_Label, 20, 255, cv.THRESH_TOZERO, img_Label)
            if np.sum(img_Label) > 5:
                return img_Label



    def randVaryROI(self,ROI):


        return ROI

    # def randMoveROI(self,ROI):
    #     #求图像的域的大小
    #     Height_Domain =  self.height_Img
    #     Width_Domain= self.width_Img
    #     #求ROI区域的坐标
    #     Rows,Cols = np.nonzero(ROI)
    #     #求ROI区域的外接矩形大小
    #     Width_ROI=np.max(Cols)-np.min(Cols)
    #     Height_ROI=np.max(Rows)-np.min(Rows)
    #     #随机设置ROI的起始坐标
    #     Row_Upleft=random.randint(0,Height_Domain-Height_ROI-1)
    #     Col_Upleft = random.randint(0, Width_Domain - Width_ROI-1)
    #     Rows=Rows-np.min(Rows)+Row_Upleft
    #     Cols=Cols-np.min(Cols)+Col_Upleft
    #     ROI_new=np.zeros([Height_Domain,Width_Domain])
    #     ROI_new[Rows,Cols]=1
    #     return ROI_new

    # def genRandImg(self,size):
    #     mean=random.randint(-125,125)
    #     fluct=random.randint(1,100)
    #     low=mean-fluct  #+(mean-fluct<0)*abs(mean-fluct)
    #     high=mean+fluct   #-(mean+fluct>255)*abs(255-(mean+fluct))
    #     img=np.random.randint(low,high,size)
    #     img=img.astype(np.float)
    #     #
    #     return img





IMG_SIZE=128
class RepairDataset(BaseDataset):
    """
    """

    def __init__(self, opt,phase="train"):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self,opt)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_size = len(self.A_paths)  # get the size of dataset A    # get the number of channels of output image
        self.trans=self.get_transform()
        self.trans_img = self.get_transform_img()
        self.trans_mask = self.get_transform_mask()
        # ztb1数据换为128, 128
        self.defectGen =DefectiveGenerator("/home/gdut/disk/lrh/Datasets/datasets/masks", (128, 128))
        self.phase=opt.phase

    def __getitem__(self, index):
        img_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        img = Image.open(img_path).convert('L')

        if self.phase == "train":
            img = self.trans(img)
            if random.random()> 0.5:
                img_defect, mask = self.defectGen.apply((np.array(img)))
                img_defect = Image.fromarray(img_defect)
                img= self.trans_img(img)
                # print(1,img.shape)
                img_defect = self.trans_img(img_defect)
                # print(2, img_defect.shape)
                mask = self.trans_mask(mask*255)
                # print(3, mask.shape)
                return {'A':img, 'B': img_defect,'C':mask, 'A_paths': img_path}
            else:
                mask = np.zeros_like(img)
                img= self.trans_img(img)
                mask = self.trans_mask(mask)
                # print(4,mask.shape)
                return {'A': img, 'B': img, 'C': mask, 'A_paths': img_path}
        else:
            img = self.trans_img(img)
            return {'A': img, 'B': img,'C': img, 'A_paths': img_path}
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size


    def get_transform(self):
        import torchvision.transforms as transforms
        l=[]
        # l.append(transforms.RandomHorizontalFlip())
        # l.append(transforms.RandomVerticalFlip())
        l.append(transforms.RandomCrop(128,padding=None))
        l.append(transforms.Resize([128, 128]))
        l.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
       # l.append(transforms.RandomResizedCrop( 256, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)))
        return   transforms.Compose(l)


    def get_transform_img(self):
        transform_list = [transforms.ToTensor()]
        grayscale=True
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transform_list)
    def get_transform_mask(self):
        return transforms.ToTensor()

class RepairDataset_valid(RepairDataset):
    """
    """

    def __init__(self,dataroot,phase,max_dataset_size):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.dir_A = dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_size = len(self.A_paths)  # get the size of dataset A    # get the number of channels of output image
        self.trans=self.get_transform()
        self.trans_img = self.get_transform_img()
        self.trans_mask = self.get_transform_mask()
        # ztb1数据换为128, 128
        self.phase=phase
    def __getitem__(self, index):
        img_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        img = Image.open(img_path).convert('L')


        img=self.trans(img)
        img = self.trans_img(img)
        return {'patch_defect': img, 'patch': img, 'mask': np.zeros_like(img),  'img_path': img_path}

    def get_transform(self):
        import torchvision.transforms as transforms
        l = []
        l.append(transforms.Resize([IMG_SIZE, IMG_SIZE]))
        return transforms.Compose(l)


def plt_show_imgs(imgs,title=None,):
    assert isinstance(imgs,(list,tuple))
    plt.figure()
    length=len(imgs)
    for i in range(length):
        plt.subplot(1,length,i+1)
        plt.imshow(imgs[i])
       # plt.imshow(imgs[i],cmap="gray")
    plt.show()

class PatchBD(object):

    def load_img(self,img_path):
        img=Image.open(img_path).convert('L')
        # return img
        # return img.crop(box=(600, 50, 7700, 2000))
        return img.crop(box=(3000, 50, 5000, 2000))

    def get_patches_per_img(self,idx,h_img, w_img, h_patch, w_patch, h_step, w_step,
                          num_patches_per_img):
        """
        :param idx:  输入图像的唯一标识
        :param h_img:  输入图像的高度
        :param w_img:  输入图像的宽度
        :param h_patch:  图像块的高度
        :param w_patch:  图像块的宽度
        :param h_step:    高度方向步长
        :param w_step:      宽度方向步长
        :param num_patch_each_img:  随机采样多少个Patch(不放回)，如果Patch不足则不采样
        :return:  Patches: list(patch)   patch=(x,y,w_patch,h_patch)
        """

        X_limit, Y_limit = w_img - w_patch, h_img - h_patch
        Xs, Ys = range(0, X_limit + 1, w_step), range(0, Y_limit + 1, h_step)
        # print(Xs)
        patches = list()
        for x in Xs:
            for y in Ys:
                patch = (x, y, w_patch, h_patch,idx)
                assert x + w_patch <= w_img, y + h_patch <= h_img
                patches.append(patch)
        # print(len(patches))
        if (len(patches) > num_patches_per_img):
            patches = random.sample(patches, num_patches_per_img)
        return patches


    def list_folder(self,root, use_absPath=True, func=None):
        """
        :param root:  文件夹根目录
        :param func:  定义一个函数，过滤文件
        :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
        :return:
        """
        root = os.path.abspath(root)
        if os.path.exists(root):
            print("遍历文件夹【{}】......".format(root))
        else:
            raise Exception("{} is not existing!".format(root))
        files = []
        # 遍历根目录,以及子目录
        for cul_dir, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(cul_dir, fname)  # .replace('\\', '/')
                if func is not None and not func(path):
                    continue
                if use_absPath:
                    files.append(path)
                else:
                    files.append(os.path.relpath(path, root))
        print("    find {} file under {}".format(len(files), root))
        return files



from ddlib.defect_genetor import DefectGenerator
class PatchData(PatchBD):

    def __init__(self,img_dir, **kwargs):
        self.img_dir = img_dir
        self.img_paths= self.list_folder(self.img_dir)
        self.h_patch = kwargs.get("h_patch", 128)
        self.w_patch = kwargs.get("w_patch", 128)
        self.num_imgs_per_epoch = kwargs.get("num_imgs_per_epoch", 100000)
        self.num_patches_per_img = kwargs.get("num_patches_per_img", 100000)
        self.h_step=kwargs.get("h_step", list(range(self.h_patch//2,self.h_patch)))
        self.w_step=kwargs.get("w_step", list(range(self.h_patch//2,self.h_patch)))
        defect_mask_dir=kwargs.get("defect_mask_dir","/home/gdut/disk/lrh/Datasets/datasets/masks")
        self.defectGen = DefectGenerator()
        self.trans = self.get_transform()
        self.trans_img = self.get_transform_img()
        self.trans_mask = self.get_transform_mask()
        self.sample_dataset()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx_patch):

        x,y,w,h,idx_img=self.patches[idx_patch]
        image=self.imgs_dict[idx_img]
        patch=image.crop(box=(x,y,x+w,y+h))
        patch = self.trans(patch)  #数据增强
        patch_defect=patch.copy()
        mask = np.zeros_like(patch)
        if idx_patch<len(self)//2:

            # patch_defect, mask = self.defectGen.apply((np.array(patch)))
            patch_defect, mask = self.defectGen(np.array(patch))
            patch_defect = Image.fromarray(patch_defect)
        patch = self.trans_img(patch)
        # print(1,img.shape)
        patch_defect = self.trans_img(patch_defect)
        # print(2, img_defect.shape)
        mask = self.trans_mask(mask * 255)
        # print(3, mask.shape)
        item={"img_path":self.img_paths[idx_img],
              "bbox":(x,y,w,h),
              "patch":patch,
              "patch_defect":patch_defect,
              "mask":mask,
              }
        return item

    def get_transform(self):
        import torchvision.transforms as transforms
        l = []
        l.append(transforms.RandomHorizontalFlip())
        l.append(transforms.RandomVerticalFlip())
        l.append(transforms.Resize([IMG_SIZE, IMG_SIZE]))
        l.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        # l.append(transforms.RandomResizedCrop( 256, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)))
        return transforms.Compose(l)

    def get_transform_img(self):
        transform_list = [transforms.ToTensor()]
        grayscale = True
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transform_list)

    def get_transform_mask(self):
        return transforms.ToTensor()


    def sample_dataset(self):
        idxes=list(range(len(self.img_paths)))
        # print(self.num_patches_per_img)
        if (len(idxes)>self.num_imgs_per_epoch):
            idxes=random.sample(idxes,self.num_imgs_per_epoch)
        self.imgs_dict = {img_idx:self.load_img(self.img_paths[img_idx]) for img_idx in idxes}
        print("从新选择{}张图片".format(len(self.imgs_dict)))
        # 从新进行图像块采样
        # print(len(idxes))
        self.patches = []
        #选择采样的patch大小和，
        h_step,w_step=random.choice(self.h_step), random.choice(self.w_step)
        for idx,img in self.imgs_dict.items():
            w_img, h_img = img.width, img.height
            patches_per_img = self.get_patches_per_img(idx,h_img, w_img,
                                                       self.h_patch,self.w_patch,
                                                       h_step,w_step,
                                                       self.num_patches_per_img)
            self.patches += patches_per_img
        print("图像分块，步长({},{}),并随机选择{}张".format(h_step,w_step,len(self.patches)))
        self.patches=self.patches+self.patches
        print("double the patches...")
class PatchDataset(BaseDataset,PatchData):

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        data_dir = opt.dataroot
        kw=dict(
            defect_mask_dir=opt.defect_mask_dir,
            num_imgs_per_epoch=opt.num_imgs_per_epoch,
            num_patches_per_img=opt.num_patches_per_img,

        )
        PatchData.__init__(self, data_dir,**kw)

    def __len__(self):
        return PatchData.__len__(self)
    def __getitem__(self, item):
        return PatchData.__getitem__(self,item)


if __name__ == '__main__':
    # data=PatchData("E:\ZTB\复赛数据\data\data\OK_Orgin_Images\OK_Orgin_Images\part1",
    #                num_imgs_per_epoch=1,num_patches_per_img=100)
    # print(len(data))
    # img=list(data.imgs_dict.values())[0]
    # img1=np.zeros_like(img)
    # for item in data:
    #     x,y,w,h=item["bbox"]
    #     img1[y:y+h,x:x+w]=1
    #
    #
    # plt_show_imgs([img,img1])
    name=r"a\abc"
    name=name.split('\\')
    print(name)
