








export CUDA_VISIBLE_DEVICES=1

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 4000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch


python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 5000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 6000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 7000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 8000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 9000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 10000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 11000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 12000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch

python infer.py --dataroot ./datasets/fabric --direction AtoB --model pix2pix --name fabric_pix2pix  \
 --netG unet_256 --input_nc 1 --output_nc 1 --no_flip  --dataset_mode repair --preprocess resize   --no_flip --eval --residual\
 --load_iter 13000 --norm instance   --checkpoints_dir   ./checkpoints/2020-03-28-06-56-02 --norm  batch