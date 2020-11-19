
import cv2
import numpy as np
from PIL import  Image


class DefectDetector(object):

    def __init__(self,**kwargs):
        self.thred_dyn =kwargs["thred_dyn"]
        self.ksize_dyn =kwargs["ksize_dyn"]
        self.ksize_close = kwargs["ksize_close"]
        self.ksize_open = kwargs["ksize_open"]
        self.thred_residual=kwargs.get("thred_residual",15)


    @staticmethod
    def cv_dyn_threshold(img, thred=15, ksize=21):
        img_blur = cv2.blur(img, ksize=(ksize, ksize))
        arr_blur = np.array(img_blur, dtype=np.float)
        arr = np.array(img, dtype=np.float)
        mask = np.where(arr - arr_blur > thred, 1, 0)
        return mask.astype(np.uint8)

    @staticmethod
    def high_pass_fft(img, filter_size=None, power_thred=None):
        assert filter_size != None or power_thred != None
        if (filter_size != None and power_thred != None):
            raise Exception("filter_size and power_thred are incompatible!")
        img_float32 = np.float32(img)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        # 将低频信息转换至图像中心
        dft_shift = np.fft.fftshift(dft)
        if power_thred != None:
            # # 获取图像尺寸 与 中心坐标
            features = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) / np.sqrt(img.shape[0] * img.shape[1])
            mask = np.where(features > power_thred, 1, 0)[:, :, np.newaxis]
        if filter_size != None:
            crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)  # 求得图像的中心点位置
            mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
            mask[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 1
        # 掩码与傅里叶图像按位相乘  去除低频区域
        fshift = dft_shift * mask  #
        # 之前把低频转换到了图像中间，现在需要重新转换回去
        f_ishift = np.fft.ifftshift(fshift)
        # 傅里叶逆变换
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back)) * 255
        return mask[:, :, 0], img_back

    @staticmethod
    def concatImage(images, mode="Adapt", scale=0.5, offset=None):
        """
        :param images:  图片列表
        :param mode:     图片排列方式["Row" ,"Col","Adapt"]
        :param scale:
        :param offset:    图片间距
        :return:
        """
        if not isinstance(images, list):
            raise Exception('images must be a  list  ')
        if mode not in ["Row", "Col", "Adapt"]:
            raise Exception('mode must be "Row" ,"Adapt",or "Col"')
        images = [np.uint8(img) for img in images]  # if Gray  [H,W] else if RGB  [H,W,3]
        images = [img.squeeze(2) if len(img.shape) > 2 and img.shape[2] == 1 else img for img in images]
        count = len(images)
        img_ex = Image.fromarray(images[0])
        size = img_ex.size  # [W,H]
        if mode == "Adapt":
            mode = "Row" if size[0] <= size[1] else "Col"
        if offset is None: offset = int(np.floor(size[0] * 0.02))
        if mode == "Row":
            target = Image.new(img_ex.mode, (size[0] * count + offset * (count - 1), size[1] * 1), 100)
            for i in range(count):
                image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
                target.paste(image, (i * (size[0] + offset), 0))
            # target.paste(image, (i * (size[0] + offset), 0, i * (size[0] + offset) + size[0], size[1]))
            return target
        if mode == "Col":
            target = Image.new(img_ex.mode, (size[0], size[1] * count + offset * (count - 1)), 100)
            for i in range(count):
                image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
                target.paste(image, (0, i * (size[1] + offset)))
            # target.paste(image, (0, i * (size[1] + offset), size[0], i * (size[1] + offset) + size[1]))
            return target

    @staticmethod
    def cv_open(mask, ksize=5, struct="ellipse"):
        assert struct in ["rect", "ellipse"]
        if struct == "rect": struct = cv2.MORPH_RECT
        if struct == "ellipse": struct = cv2.MORPH_ELLIPSE
        elment = cv2.getStructuringElement(struct, (ksize, ksize))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, elment)
        return mask_open

    @staticmethod
    def cv_close(mask, ksize=5, struct="ellipse"):
        assert struct in ["rect", "ellipse"]
        if struct == "rect": struct = cv2.MORPH_RECT
        if struct == "ellipse": struct = cv2.MORPH_ELLIPSE
        elment = cv2.getStructuringElement(struct, (ksize, ksize))
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, elment)
        return mask_open

    @staticmethod
    def to255(array,flag=1):
        if flag==1:
            array= array * 127.5 + 127.5
        elif flag==0:
            array= array * 255
        else:
            raise Exception("got an wrong param “flag")
        return np.array(array,np.uint8)



    def detect_on_image(self,img):
        mask=self.cv_dyn_threshold(img,thred=self.thred_dyn,ksize=self.ksize_dyn)
        close_mask = self.cv_close(mask, ksize=self.ksize_close)
        open_mask = self.cv_open(close_mask, ksize=self.ksize_open)
        num_rois, rois = cv2.connectedComponents(open_mask)
        ROI = np.zeros_like(mask)
        if len(rois) == 0:
            return ROI , mask
        for roi_idx in range(1, num_rois):
            # 求ROI区域的坐标
            Cols, Rows = np.nonzero(np.where(rois == roi_idx, 1, 0))
            # 求ROI区域的外接矩形大小
            h1, h2, w1, w2 = np.min(Cols), np.max(Cols), np.min(Rows), np.max(Rows)
            ROI[h1:h2, w1:w2] = 1
        #return mask,close_mask,open_mask,ROI, mask*ROI
        return  ROI, mask * ROI
    def detect_on_batch(self,array):
        num=array.shape[0]
        imgs=[ np.array(array[idx],np.uint8).squeeze() for idx in range(num) ]
        results=[self.detect_on_image(img) for img in  imgs]
        lens=len(results[0])
        re=list()
        for i in range(lens):
            item= [  self.to255(item[i],0) for item in results]
            re.append(np.array(item))
        return re


    def residual(self,batch1, batch2,norm=True,mean=True):
        """
        求输入图片和重建图片的差，
        小于thred_residual的过滤
        :return:
        """

        batch1 = batch1.astype(np.float)
        batch2 = batch2.astype(np.float)
        # re = np.abs(batch1 - batch2)
        re = np.abs(batch2 - batch1)
        if mean:
            mean=np.mean(re)
            # print(mean)
            re=np.abs(re-mean)
        re[np.where(re<self.thred_residual/255.)]=0
        max_val = np.max(re, axis=(1, 2, 3), keepdims=True)
        min_val = np.min(re, axis=(1, 2, 3), keepdims=True)
        if norm:
            return (re - min_val) / (max_val - min_val)
        else:
            return re


    def apply(self,imgs_batch,labels_batch,reconst_batch,with_label=False):
        res_batch=self.residual(imgs_batch,reconst_batch)
        imgs_batch=self.to255(imgs_batch,1)
        if with_label:
            labels_batch = self.to255(labels_batch, 0)
        reconst_batch = self.to255(reconst_batch, 1)

        res_batch = self.to255(res_batch, 0)
        # res_batch1=np.where(res_batch>10,255,0)
        item_batchs=self.detect_on_batch(res_batch)
        if with_label:
            return [imgs_batch,labels_batch,reconst_batch,res_batch]+item_batchs
        else:
            return [imgs_batch, reconst_batch, res_batch] + item_batchs



    def thre(self,map_batch,thre=125,norm=False):
        map_image=map_batch.astype(np.float)
        re=np.where(map_image<thre/255.,0,1)
        max_val = np.max(re, axis=(1, 2, 3), keepdims=True)
        min_val = np.min(re, axis=(1, 2, 3), keepdims=True)
        if norm:
            return (re - min_val) / (max_val - min_val)
        else:
            return re

    def ztb_rangd2_apply(self,image_batch,labels_batch,map_batch,with_label=False):
        if with_label:
            labels_batch = self.to255(labels_batch, 0)
        imgs_batch=self.to255(image_batch,1)
        rebuild_images=self.to255(map_batch,1)

        res_batch=self.thre(map_batch,thre=15)
        res_batch=self.to255(res_batch,0)

        item_batchs = self.detect_on_batch(res_batch)

        if with_label:
            return [imgs_batch,labels_batch,rebuild_images,res_batch]+item_batchs
        else:
            return [imgs_batch, rebuild_images, res_batch] + item_batchs





    # def apply(self,imgs_batch,labels_batch,reconst_batch,with_label=False):
    #     res_batch=self.residual(imgs_batch,reconst_batch)
    #     imgs_batch=self.to255(imgs_batch,1)
    #     if with_label:
    #         labels_batch = self.to255(labels_batch, 0)
    #     reconst_batch = self.to255(reconst_batch, 1)
    #
    #     res_batch = self.to255(res_batch, 0)
    #     # res_batch1=np.where(res_batch>10,255,0)
    #     item_batchs=self.detect_on_batch(res_batch)
    #     if with_label:
    #         return [imgs_batch,labels_batch,reconst_batch,res_batch]+item_batchs
    #     else:
    #         return [imgs_batch, reconst_batch, res_batch] + item_batchs
