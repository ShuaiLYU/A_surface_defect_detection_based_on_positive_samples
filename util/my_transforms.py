from torchvision import  transforms
import numbers
import  collections
import random
from torchvision.transforms import functional as F
import torch
from PIL import Image
import  numpy as np
import math



def t2n(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

class Tensor2Numpy(object):
	def __call__(self, imgs):
		def func(x):
			if isinstance(x, torch.Tensor):
				x = x.cpu().detach().numpy()
			return x
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class To3DNumpy(object):
	def __init__(self,order):
		assert order in ("hwc","chw")
		self.order=order
	def __call__(self, imgs):
		def func(img):
			img=np.array(img)
			if img.ndim==3:
				if self.order=="hwc":
					return img
				elif self.order=="chw":
					return img.transpose(2,0,1)
			elif img.ndim == 2:
				if self.order=="hwc":
					return img[:, :, np.newaxis]
				elif self.order=="chw":
					return img[np.newaxis,:, :]
			else:
				return Exception("input's ndim must be 2 or 3!!!")
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs


class ToFloatNumpy(object):
	def __call__(self, imgs):
		def func(img):
			return np.array(img).astype(np.float)
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class ToInt8Numpy(object):
	def __call__(self, imgs):
		def func(img):
			return np.array(img).astype(np.uint8)
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class ToUniform(object):

	def __call__(self, imgs):
		#为止错误
		def func(img):
			img = img.astype(np.uint8)
			print(img.dtype)
			img = img/255.0
			return img
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class ToMask(object):
	def __init__(self,thred=0):
		super(ToMask,self).__init__()
		self.thred=thred
	def __call__(self, imgs):
		def func(img):
			if isinstance(img,torch.Tensor):
				return torch.where(img>self.thred,torch.full_like(img,1),torch.full_like(img,0)).int()
			return np.where(np.array(img)>self.thred,1,0).astype(np.int8)
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class ToPIL(object):
	def __call__(self, imgs):

		def func(img):
			if  isinstance(img, np.ndarray) and img.ndim == 2:
				# if 2D image, add channel dimension (HWC)
				img = np.expand_dims(img, 2)
			return  F.to_pil_image(img)
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		# for img in imgs:
		# 	#print(img.size)
		# 	print(np.array(img).shape)
		return imgs



class ToLong(object):
    def __call__(self, x):
        return torch.LongTensor(np.asarray(x))

class GroupResize(transforms.Resize):
	"""Resize the input PIL Image to the given size.

	Args:
	    size (sequence or int): Desired output size. If size is a sequence like
	        (h, w), output size will be matched to this. If size is an int,
	        smaller edge of the image will be matched to this number.
	        i.e, if height > width, then image will be rescaled to
	        (size * height / width, size)
	    interpolation (int, optional): Desired interpolation. Default is
	        ``PIL.Image.BILINEAR``
	"""

	def __call__(self, imgs):
		"""
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
		def func(img):
			return  F.resize(img, self.size, self.interpolation)
		if isinstance(imgs, collections.Iterable):
			for idx in range(len(imgs)):
				imgs[idx]=func(imgs[idx])
		else:
			imgs=func(imgs)
		return imgs

class GropuRandomCropAndScale(transforms.RandomResizedCrop):
	def __init__(self,area_limit=1e+8, size_ratio=(3. / 4., 4. / 3.), crop_scale=(0.5, 1.0), crop_ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):

		self.interpolation = interpolation
		self.crop_scale = crop_scale
		self.crop_ratio = crop_ratio
		self.size_ratio=size_ratio
		self.area_limit=area_limit

	@staticmethod
	def get_params(img, scale, ratio):
		"""Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
		#宽高比
		img_ratio=img.size[0]/img.size[1]
		img_area=img.size[0]*img.size[1]
		for attempt in range(10):
			target_area = random.uniform(*scale) * img_area
			aspect_ratio = random.uniform(*ratio) * img_ratio

			w = int(round(math.sqrt(target_area * aspect_ratio)))
			h = int(round(math.sqrt(target_area / aspect_ratio)))


			if w <= img.size[0] and h <= img.size[1]:
				i = random.randint(0, img.size[1] - h)
				j = random.randint(0, img.size[0] - w)
				return i, j, h, w

		# Fallback
		# w = min(img.size[0], img.size[1])
		# i = (img.size[1] - w) // 2
		# j = (img.size[0] - w) // 2
		return 0,0,img.size[0],img.size[1]

	def get_output_size(self,input_size,size_ratio,area_limit):
		img_ratio = input_size[0] / input_size[1]
		img_area = input_size[0] * input_size[1]
		target_area = random.uniform(*size_ratio) * img_area
		target_area = min(target_area,area_limit)
		w = int(round(math.sqrt(target_area * img_ratio)))
		h = int(round(math.sqrt(target_area / img_ratio)))
		out_size=(w,h)
		return out_size
	def __call__(self, imgs):
		"""
		Args:
			img (PIL Image): Image to be cropped and resized.

		Returns:
			PIL Image: Randomly cropped and resized image.
		"""
		i, j, h, w = self.get_params(imgs[0], self.crop_scale, self.crop_ratio)
		out_size=self.get_output_size((h,w),self.size_ratio,self.area_limit)
		imgs=[F.resized_crop(img, i, j, h, w, out_size, self.interpolation) for img in imgs]

		return imgs

class  GroupRandomRotation(transforms.RandomRotation):
	def __init__(self, degrees=[0,90,180,270], resample=False, expand=True, center=None):
		if not isinstance(degrees, list):
			raise ValueError("it must be a list.")
		self.degrees = degrees
		self.resample = resample
		self.expand = expand
		self.center = center
	@staticmethod
	def get_params(degrees):
		"""Get parameters for ``rotate`` for a random rotation.

		Returns:
			sequence: params to be passed to ``rotate`` for random rotation.
		"""
		rand = random.randint(0, len(degrees)-1)
		angle=degrees[rand]
		return angle
	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be rotated.

		Returns:
			PIL Image: Rotated image.
		"""
		angle =self.get_params(self.degrees)
		if isinstance(img,list):
			return [F.rotate(item, angle, self.resample, self.expand, self.center)
					for item in img ]
		else:
			return F.rotate(img, angle, self.resample, self.expand, self.center)

class GroupRandomHorizontalFlip(transforms.RandomHorizontalFlip):
	def __init__(self, p=0.5):
		super(GroupRandomHorizontalFlip,self).__init__(p)

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be flipped.

		Returns:
			PIL Image: Randomly flipped image.
		"""
		rand= random.random()
		if isinstance(img,list):
			img= [F.hflip(item) if rand< self.p else item for item in img ]
		else:
			img = F.hflip(img) if  rand < self.p else img
		return img


class GroupRandomVerticalFlip(transforms.RandomVerticalFlip):
	def __init__(self, p=0.5):
		super(GroupRandomVerticalFlip, self).__init__(p)

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be flipped.

		Returns:
			PIL Image: Randomly flipped image.
		"""
		rand = random.random()
		if isinstance(img, list):
			img = [F.vflip(item) if rand < self.p else item for item in img]
		else:
			img = F.vflip(img) if rand < self.p else img
		return img


class ComposeJoint(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, x):
		for transform in self.transforms:
			x = self._iterate_transforms(transform, x)

		return x

	def _iterate_transforms(self, transforms, x):
		if isinstance(transforms, collections.Iterable):
			for i, transform in enumerate(transforms):
				x[i] = self._iterate_transforms(transform, x[i])
		else:

			if transforms is not None:
				x = transforms(x)

		return x