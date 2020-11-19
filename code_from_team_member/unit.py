import os
import sys
import glob
import numpy
import json
import cv2
import random
def Image_name_modification(picture_path):
    """
    把文件夹里面的图片按照0-len(num)-1的顺序进行排序
    :param picture_path: 输入文件夹的路径
    """
    train_picture_list = glob.glob(os.path.join(picture_path, "*"))
    for i,basename in enumerate(train_picture_list):
        # image_path=os.path.join(picture_path,basename)
        basename=basename
        os.rename(basename,str(i)+".jpg")

def get_point(arr,condition):
    # point_list=[]
    point_list=numpy.argwhere(arr >= condition)
    return point_list

def write_to_json(picture_height,pitture_width,picture_name,region,save_name):
    # json_dict={}
    json_dict={
        "Height": picture_height,
        "Width": pitture_width,
        "name": "{}".format(picture_name)+'bmp',
        "regions":region,
    }
    e=json.dumps(json_dict)
    with open('{}.json'.format(save_name),'w',encoding='utf-8',) as w:
        w.write(e+'\n')



class submit(object):
    def __init__(self,picture_height,pitture_width,picture_name,picture,save_path='result/data/focusight1_round1_train_part1/TC_Images',picture_model='BIN'):
        '''
        :param picture_height: 写入jison文件的图片的高
        :param pitture_width:写入jison文件图片的宽
        :param picture_name: 图片名字
        :param picture: 输入的图片
        :param picture_model: 图片输入的模式，有灰度模式和二值化模式 BIN
        '''
        super(submit, self).__init__()
        self.picture_height=picture_height
        self.pitture_width=pitture_width
        self.picture_name=picture_name
        self.picture_dim=picture.shape[-1]
        # print(self.picture_dim)
        self.picture=picture if picture.shape==(self.picture_height,self.pitture_width,self.picture_dim)  else self.resize(picture)
        # print('Size of image is :{}'.format(self.picture.shape))
        self.picture_model=picture_model
        self.x=self.get_point(self.picture)
        self.save_path=save_path
        #区域的大小进行筛选，正样本和非常小的误差点将不会写进jison
        self.region_thre=0
        if len(self.x["points"])>self.region_thre:
            # print('write to jison')
            self.write_to_json(self.picture_height,self.pitture_width,self.picture_name,self.x,save_name=self.picture_name,save_path=self.save_path)


    def resize(self,image):
        img=cv2.resize(image,(self.picture_height,self.pitture_width))
        return img

    def Connected_domain(self,picture):
        pass

    def get_point(self,picture):
        self.point_dict={}
        if self.picture_model=='BIN':
            # print(picture)
            point = numpy.argwhere(picture > 100)
            # print(point)
            point_finally = numpy.array([[i[1], i[0]] for i in point])
            # print(point_finally)
            # x = [(str(i).split('[')[-1].split(']')[0]).lstrip().rstrip().split(' ')[0] + ',' +
            #      (str(i).split('[')[-1].split(']')[0]).lstrip().rstrip().split(' ')[-1] for i in point_finally]
            x = [(str(i).split('[')[-1].split(']')[0]).lstrip().rstrip().split(' ')[-1] + ',' +
                 (str(i).split('[')[-1].split(']')[0]).lstrip().rstrip().split(' ')[0] for i in point_finally]
            self.point_dict["points"]=x
            return self.point_dict
        else:
            pass

    def write_to_json(self,picture_height, pitture_width, picture_name, region, save_name,save_path):
        # json_dict={}
        json_dict = {
            "Height": picture_height,
            "Width": pitture_width,
            "name": "{}.bmp".format(picture_name),
            "regions": [region],
        }
        e = json.dumps(json_dict)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open('./{}/{}.json'.format(save_path,save_name), 'w', encoding='utf-8', ) as w:
            w.write(e + '\n')


def main():
    # train_picture_path='./dataset_commutator/train_picture/'
    # train_picture_lable='./dataset_commutator/train_label/'
    # #Image_name_modification(train_picture_path)
    # Image_name_modification(train_picture_lable)
    # import numpy
    # a = numpy.array(([3, 2, 1], [2, 5, 7], [4, 7, 8]))
    # itemindex = numpy.argwhere(a >= 7)
    # print(a)
    # print(itemindex)
    # print(itemindex[1])
    # print(a)


  #   write_to_json(128,128,"0kfkmg89WfJ308GZY3EQ850AtO28YT.bmp",[
  #   {
  #     "points":[
  #       "13, 86","14, 85","14, 86","14, 87","15, 84","15, 85","15, 86"
  #     ]
  #   },
  #   {
  #     "points":[
  #       "24, 84","24, 85","24, 86","24, 87","24, 88","24, 89","25, 85","25, 86"
  #     ]
  #   }
  # ],save_name="0kfkmg89WfJ308GZY3EQ850AtO28YT.bmp")
    picture=cv2.imread('./00b5CacodX4rVA9jmo6aad56xvwMT3.jpg',0)
    submit(128,128,'00b5CacodX4rVA9jmo6aad56xvwMT3',picture)
    # print(point_finally)
    # detecture=submit(128,128,'groundT_Bbox_bn_d3.bmp',picture)

if __name__=="__main__":
    main()
