

python train.py --dataroot  /media/host/新加卷/ls_ZTB/gen/trainA2_yuan\
      --dataset_mode patch  --defect_mask_dir /media/host/新加卷/征图杯１/datasets/masks \
       --num_imgs_per_epoch 50  --num_patches_per_img 100000000   --max_dataset_size_valid  512 \
       --name ztb  --model segment  --weight_decay_S 0.0001  --lr_S 0.001 \
       --direction AtoB   \
       --net_S unet_256   --input_nc 1 --output_nc 1    --norm  instance \
       --batch_size 80 --n_epochs 200  --preprocess resize   \
       --checkpoints_dir  ./checkpoints/checkpoints-segment \
       --dataroot_valid /media/host/新加卷/征图杯/datasets/fabric/testA2

