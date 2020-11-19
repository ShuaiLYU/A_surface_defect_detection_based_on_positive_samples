# coding=UTF-8
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import  DefectDetector
from code_from_team_member.unit import submit
import cv2
import zipfile
import time

def vaild(epoch):
    opt = TestOptions().parse()  # get test options
    opt.eval=True
    opt.load_iter=epoch
    opt.checkpoints_dir='./checkpoints/checkpoints-runzt'
    opt.name='fabric_pix2pix'
    opt.dataroot='../datasets/fabric'
    opt.direction='AtoB'
    opt.model='pix2pix'
    opt.netG='unet_128'
    opt.input_nc=1
    opt.output_nc=1
    opt.dataset_mode='repair'
    opt.preprocess='resize'
    opt.residual=True
    opt.norm='instance'
    opt.n_epochs=3000
    # opt.lr_policy='cosine'
    opt.continue_train=True
    # print('!!!!!!!!!!!')
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    kwargs = {
        "thred_dyn": 50,
        "ksize_dyn": 100,
        "ksize_close": 30,
        "ksize_open": 3,
    }
    detector = DefectDetector(**kwargs)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        print(i)
        paths = data["A_paths"]
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        image_batch = visuals["real_A"]
        reconst_batch = visuals["fake_B"]
        # B_mask=data["B_mask"]

        image_batch = image_batch.detach().cpu().numpy()
        # label_batch = B_mask.detach().cpu().numpy()
        reconst_batch = reconst_batch.detach().cpu().numpy()
        # batchs=detector.apply(image_batch,label_batch,reconst_batch)
        batchs = detector.apply(image_batch, image_batch, reconst_batch)
        for idx, path in enumerate(paths):
            visual_imgs = []
            for batch in batchs:
                visual_imgs.append(batch[idx].squeeze())
            img_visual = detector.concatImage(visual_imgs, offset=None)
            # print(img_visual.size)
            visualization_dir = opt.checkpoints_dir + "/infer_epoch{}/".format(opt.load_iter)
            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir)
            img_visual.save(visualization_dir + "_".join(path.split("/")[-2:]))


def infer_model(model,dataset,epoch):
    model.eval()
    for i, data in enumerate(dataset):
        paths = data["A_paths"]
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        image_batch = visuals["img"]
        mask_batch = visuals["mask"]
        reconst_batch = visuals["pred"]
        image_batch = image_batch.detach().cpu().numpy()
        mask_batch = mask_batch.detach().cpu().numpy()
        reconst_batch = reconst_batch.detach().cpu().numpy()
        for idx, path in enumerate(paths):
            visual_imgs = []
            for batch in [image_batch,mask_batch,reconst_batch]:
                visual_imgs.append(batch[idx].squeeze())
            img_visual = detector.concatImage(visual_imgs, offset=None)
            visualization_dir = opt.checkpoints_dir + "/infer_epoch{}/".format(epoch)
            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir)
            img_visual.save(visualization_dir + "_".join(path.split("/")[-2:]))
    model.train()


def make_zip(source_dir, output_filename):
  zipf = zipfile.ZipFile(output_filename, 'w')
  pre_len = len(os.path.dirname(source_dir))
  for parent, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
      pathfile = os.path.join(parent, filename)
      arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
      zipf.write(pathfile, arcname)
  zipf.close()

def main():


    opt = TestOptions().parse()  # get test options
    opt.netG = 'unet_128'
    gen_jison = False
    if gen_jison:
        vis = False
        opt.num_test=50000
    else:
        vis = True
        opt.num_test = 300
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    print(model.__dict__)
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    kwargs = {
        "thred_dyn": 50,
        "ksize_dyn": 100,
        "ksize_close": 30,
        "ksize_open": 3,
    }
    detector = DefectDetector(**kwargs)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        # print(i)
        paths = data["A_paths"]
        print(paths)
        # print(len(paths))
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        imgee_show=cv2.imread(paths[0])
        # print(visuals["pre_score"])
        cv2.imshow('image',imgee_show)
        cv2.waitKey()
        image_batch = visuals["real_A"]
        reconst_batch = visuals["fake_B"]
        # B_mask=data["B_mask"]

        image_batch = image_batch.detach().cpu().numpy()
        # label_batch = B_mask.detach().cpu().numpy()
        reconst_batch = reconst_batch.detach().cpu().numpy()
        # batchs=detector.apply(image_batch,label_batch,reconst_batch)
        batchs = detector.ztb_rangd2_apply(image_batch, image_batch, reconst_batch)



        image_high=128
        image_width=128
        image_name=paths[0].split('/')[-1].split('.')[0]
        image=batchs[-1].squeeze()
        # print(image.shape)
        # cv2.imshow('image',image)
        # cv2.waitKey(3)
        submit(image_high,image_width,image_name,image)
        if vis:
            for idx, path in enumerate(paths):
                visual_imgs = []
                for batch in batchs:
                    visual_imgs.append(batch[idx].squeeze())
                img_visual = detector.concatImage(visual_imgs, offset=None)
                # print(img_visual.size)
                visualization_dir = opt.checkpoints_dir + "/infer_epoch{}/".format(opt.load_iter)
                if not os.path.exists(visualization_dir):
                    os.makedirs(visualization_dir)
                img_visual.save(visualization_dir + "_".join(path.split("/")[-2:]))
    print(i)


if __name__ == '__main__':
    start = time.time()
    main()
    make_zip('./result/', './result/data.zip')
    end = time.time()
    print(end-start)
