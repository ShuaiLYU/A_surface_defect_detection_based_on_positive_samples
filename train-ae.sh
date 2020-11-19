


python train.py --dataroot  /media/host/新加卷/ls_ZTB/gen/trainA2_yuan\
      --dataset_mode patch  --defect_mask_dir /media/host/新加卷/征图杯１/datasets/masks \
       --num_imgs_per_epoch 20  --num_patches_per_img 100000000   --max_dataset_size_valid  512 \
         --model ae    --weight_decay 0.0001  --lr 0.001 \
       --direction AtoB   \
       --batch_size 512 --n_epochs 200  --preprocess resize   \
         --checkpoints_dir  ./checkpoints/checkpoints-ae --name ztb-stage2-ae \
       --dataroot_valid /media/host/新加卷/征图杯/datasets/fabric/testA2

