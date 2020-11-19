
import  numpy as np
from PIL import  Image


"""
shape=(img_h, img_w)
size=(img_w,img_h)
radian  弧度
angle   角度
"""



def _is_pil_image(img):
        return isinstance(img, Image.Image)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    """
    判读numpy是否是图片，
    并且图片的通道在最后一个维度(不是一个严谨的判断)
    :param img:
    :return:
    """
    return img.ndim==2 or (img.ndim==3 and img.shape[2]<5)

def get_img_shape(img):
    if _is_pil_image(img):
        return img.size[::-1]

    elif _is_numpy_image(img):
        return img.shape[:2]
    else:
        raise Exception("got an upexpected input!")


from collections.abc import Iterable
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)



def check_size(size):
    assert isinstance(size, int) or (issubclass(size, Iterable) and len(size) == 2)

import os
def list_folder(root, use_absPath=True, func=None):
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
