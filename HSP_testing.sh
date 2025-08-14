#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=fruit_testing_job_20250814
#SBATCH --output=fruit_testing_job_20250814.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1528328@utah.edu


# Run the testing script
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/objects_4-24/fruit/processed_fruit_4-24 --name hyperspectral_fruit_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

#python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
