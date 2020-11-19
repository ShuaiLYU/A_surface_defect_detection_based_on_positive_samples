
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from  ddlib.utils import  *
import math
import  random


class RandomRotate(object):

    def __init__(self,angle=(0,360),probability=1):
        self.angle=angle
        self.probability=probability
        assert 0<=probability<=1
        assert isinstance(angle,Iterable)
    @staticmethod
    def rotate(img, mask, angle):
        """
        :param img:
        :param angle:
        :param mask:
        :return:
        """
        assert np.mean(mask) > 0 and np.max(mask) < 2
        if isinstance(angle, Iterable):
            angle = np.random.randint(angle[0], angle[1])
        assert get_img_shape(img) == get_img_shape(mask),print(img.size,mask.size)
        Row,Col = np.nonzero(mask)

        # img=img.crop(box=(np.min(Col),np.min(Row),np.max(Col),np.max(Row)))
        # mask = mask.crop(box=(np.min(Col), np.min(Row), np.max(Col), np.max(Row)))
        img = np.array(img)[np.min(Row): np.max(Row) + 1, np.min(Col): np.max(Col) + 1]
        mask = np.array(mask)[np.min(Row): np.max(Row) + 1, np.min(Col): np.max(Col) + 1]

        # 扩充为正方形，防止边界信息丢失
        h, w = get_img_shape(img)
        img_rotate=np.zeros((h+w,h+w),dtype=img.dtype)
        mask_rotate=np.zeros((h+w,h+w),dtype=mask.dtype)
        img_rotate[w//2:w//2+h,h//2:h//2+w]=img
        mask_rotate[w // 2:w // 2 + h, h // 2:h // 2 + w] = mask
        # pad_left, pad_right, pad_up, pad_down = h // 2, h // 2, w // 2, w // 2
        # img = np.pad(img, ((pad_up, pad_down), (pad_left, pad_right)), mode="constant", constant_values=0)
        # mask = np.pad(mask, ((pad_up, pad_down), (pad_left, pad_right)), mode="constant", constant_values=0)
        #图像旋转
        img = F.rotate(Image.fromarray(img_rotate), angle, Image.BILINEAR)
        mask = F.rotate(Image.fromarray(mask_rotate), angle, Image.NEAREST)
        assert np.max(mask) > 0
        return img, mask

    def __call__(self, img, mask):
        """
        :param img:
        :param angle:
        :param mask:
        :return:
        """
        if random.random()>self.probability:
            return img,mask
        img,mask=self.rotate(img,mask,self.angle)
        return img, mask


class RandResize(object):
    def __init__(self,scale=None,scale_fator=(0.5,2),ratio=(0.5,2),probability=1):
        assert (scale_fator!=None and scale==None) or (scale_fator==None and scale!=None)\
            ,"scale_fator and scale  are incompatible"
        self.scale=scale
        self.scale_factor=scale_fator
        self.probability = probability
        assert 0 <= probability <= 1
        self.ratio=ratio
        assert isinstance(self.ratio, Iterable)
        if self.scale_factor !=None:
            assert isinstance(self.scale_factor, Iterable)
        if self.scale !=None:
            assert isinstance(self.scale, Iterable)
    @staticmethod
    def resize(img, mask, scale, ratio=(0.5,2)):
        w = int(scale * np.sqrt(ratio))
        h = int(scale / np.sqrt(ratio))
        if not isinstance(img,Image.Image):
            img=Image.fromarray(img)
        img=F.resize(img,(w,h),interpolation=Image.BILINEAR)
        mask = F.resize(mask, (w, h), interpolation=Image.NEAREST)
        return img, mask
    def __call__(self, img,mask):
        if random.random() > self.probability:
            return img, mask
        assert np.max(mask) > 0
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        while True:
            if self.scale!=None:
                scale=np.random.randint(self.scale[0], self.scale[1])
            else:
                scale_factor=(self.scale_factor[0] + np.random.rand() * (self.scale_factor[1] - self.scale_factor[0]))
                scale=np.sqrt(img.size[0]*img.size[1])*scale_factor
            ratio = self.ratio[0] + np.random.rand() * (self.ratio[1] - self.ratio[0])
            img_resize,mask_resize=self.resize(img,mask,scale,ratio)
            if   np.max(mask_resize)>0:
                return img_resize,mask_resize


class RandomGrayLevelTrans(object):

    def __init__(self,mean=None,mean_factor=(0.5,2),probability=1):
        assert (mean_factor != None and mean == None) or (mean_factor == None and mean != None) \
            , "scale_fator and scale  are incompatible"
        self.mean = mean
        self.mean_factor = mean_factor
        self.probability = probability
        assert 0 <= probability <= 1
        if self.mean_factor != None:
            assert isinstance(self.mean_factor, Iterable)
        if self.mean != None:
            assert isinstance(self.mean, Iterable)

    @staticmethod
    def trans(img, mask,mean):
        assert np.mean(mask) > 0 and np.max(mask) < 2
        img,mask=np.array(img),np.array(mask)
        img = img.astype(np.float)

        mean_old = np.sum(img * mask) / np.sum(mask)
        # print(mean_old,mean)
        # img = (img / mean_old * mean)
        img = (img - mean_old + mean)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img) ,Image.fromarray( mask)

    def __call__(self, img,mask):
        if random.random() > self.probability:
            return img, mask
        if self.mean!=None:
            mean=np.random.randint(self.mean[0], self.mean[1])
        else:
            scale_factor=(self.mean_factor[0] + np.random.rand() * (self.mean_factor[1] - self.mean_factor[0]))
            mean=int(np.mean(img)*scale_factor)
        img,mean=self.trans(img,mask,mean)
        return img,mean


def plt_show_imgs(imgs,title=None,):
    assert isinstance(imgs,(list,tuple))
    plt.figure()
    length=len(imgs)
    for i in range(length):
        plt.subplot(1,length,i+1)
        # plt.imshow(imgs[i])
        plt.imshow(imgs[i],cmap="gray")
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img=Image.open("E:\ZTB\复赛数据\data-part1\data\TC_Images\TC_Images\part1\TC_Images/1Apqv1IV0hBi0gjgyF5EJ9vLYE4jl6.bmp")
    roi=np.zeros_like(img)
    roi[30:50,20:80]=1

    # ranrotate=RandomRotate()
    # img,roi=ranrotate(img,roi)
    #
    ran2=RandResize(scale_fator=(0.5,2),ratio=(0.5,2))

    ran3=RandomGrayLevelTrans(mean=(10,40),mean_factor=None)
    img, roi = ran3(img, roi)
    print(np.array(img))
    plt_show_imgs([img, roi])