#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=training_job_20250814_2
#SBATCH --output=training_job_20250814_2%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1528328@utah.edu

## Activate the conda environment
source activate hsp_env

## Navigate to the project directory
cd /uufs/chpc.utah.edu/common/home/u1528328/HSP

# Run the training/testing scripts
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mkrauss/William_Summer --name hyperspectral_william_8-14 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_william_8-14 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mkrauss/fruit --name hyperspectral_fruit_8-14 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_8-14 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes --name hyperspectral_banknotes_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_8-14_v2 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0