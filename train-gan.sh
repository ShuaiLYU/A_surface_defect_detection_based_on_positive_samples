

python train.py --dataroot /media/host/新加卷/ls_ZTB/gen/trainA2_yuan\
  --dataset_mode patch  --defect_mask_dir /media/host/新加卷/征图杯１/datasets/masks \
  --num_imgs_per_epoch 20  --num_patches_per_img 100000000   --max_dataset_size_valid  512 \
       --model pix2pix --direction AtoB    \
       --netG unet_256 --netD  pixel --input_nc 1 --output_nc 1  --dataset_mode patch \
       --batch_size 128 --n_epochs 100000 --gan_mode lsgan  --preprocess resize   \
       --weight_decay_G 0.001  --lr_decay_iters 100   --norm  batch --lambda_L1 100\
       --checkpoints_dir  ./checkpoints/checkpoints-ztb-gan  --name ztb-gan\
       --dataroot_valid /media/host/新加卷/征图杯/datasets/fabric/testA2
















































