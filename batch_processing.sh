#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=64Gs
#SBATCH --gres=gpu:3090:1
#SBATCH --time=6:00:00
#SBATCH --job-name=training_job_hyperspectral_1024_large_nogan_double
#SBATCH --output=training_job_hyperspectral_1024__large_nogan_double_%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=a.ingold@utah.edu   


## Load necessary modules (if any)
## module load <module-name>

## Activate the conda environment
source activate pix2pix

## Navigate to the project directory
cd probabilistic_pix2pix_chpc

/scratch/general/nfs1/u1528328/img_dir/2-20_90degree/

/scratch/general/nfs1/u1528328/img_dir/resolution_datasets/45_degree/thorlabs

## Run the training script
##python train.py --dataroot ./datasets/combined_plant_0bp_noBlurryRemoval_pix2pi$  # hyperspectral_3D_SSIM_100_SPEC_0_degree
##python train.py --dataroot ./datasets/combined_plant_0bp_noBlurryRemoval_pix2pi$

### ------------------- LCD Model Training -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Original Model Training 
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/2-24_45degree --name hyperspectral_1024_large_nogan_double_2-20_90degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_MAE_loss --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
# Original Model Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/2-24_45degree --name hyperspectral_1024_large_nogan_double_2-24_45degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-24_45degree --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135
# Resolution Chart Model
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/2-22_0degree --name hyperspectral_3D_SSIM_SPEC_0_degree_RES_CHART --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_RES_CHART --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0

### --------------- Current Optimal Models (1x1 bottleneck) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Banknote Model Training (V2)
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_banknotes_b_no_mask_45_degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask_45_degree --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_banknotes_b_no_mask_90_degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask_90_degree --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_banknotes_b_no_mask_135_degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask_135_degree --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Pokemon Dataset Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_pokemon/augmented --name hyperspectral_pokemon_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_pokemon_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# Fruit Dataset Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit/augmented --name hyperspectral_fruit_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# Fruit Dataset V2 Testing
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_4-24/augmented/split --name hyperspectral_fruit_v2_deep_wide --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2_deep_wide --model pix2pix --input_nc 1 --output_nc 106 --ngf 128 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024_mod --netG_reps 2 --netD_mult 0 --polarization 0 

# Combined Dataset Training (Pokemon + Fruit + Banknotes)
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/combined_dataset --name hyperspectral_combined_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_combined_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# Combined Dataset Training (Pokemon + Fruit + Banknotes + Color Chart + Bugs)
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/combined_dataset --name hyperspectral_combined_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_combined_v2 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# White Resolution Chart Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-28/augmented --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# Invertebrate Dataset Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_invertebrates_4-29/augmented --name hyperspectral_invertebrates_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_invertebrates_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 


### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Train & test W-NET on multiple datasets and polarizations.
#!/bin/bash

# Geology Sample 1 Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Geology Sample 2 Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Shells and Fossils Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135


python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 --name Geology_Sample_1_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_2 --name Geology_Sample_2_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_2_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol0_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol0_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol45_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol45_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol90_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol90_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Shells_and_Fossils --name Shells_and_Fossils_pol135_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Shells_and_Fossils_pol135_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135


# Spectral Resolution Chart Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-30/spectro-pol/3900K  --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-30/spectro-pol/3900K  --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-30/spectro-pol/3900K  --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-30/spectro-pol/3900K  --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/USAF/spectro_pol --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/USAF/spectro_pol --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/USAF/spectro_pol --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/USAF/spectro_pol --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135


# White Resolution Chart Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_res_charts_4-28/augmented  --name hyperspectral_res_charts_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_res_charts_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Invertebrate Dataset Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_invertebrates_4-29/augmented --name hyperspectral_invertebrates_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_invertebrates_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0


# Banknote Model Testing - Polarizer
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/pol_sorted/0_degree --name hyperspectral_banknotes_v3 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/pol_sorted/45_degree --name hyperspectral_banknotes_v3_45 --checkpoints_dir /scratch/general/nfs1/users/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_45 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/pol_sorted/90_degree --name hyperspectral_banknotes_v3_90 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_90 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/pol_sorted/135_degree --name hyperspectral_banknotes_v3_135 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_135 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Banknote Polarizer Videos
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/objects_4-16/processed_videos/hs --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/objects_4-16/processed_videos/hs --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/objects_4-16/processed_videos/hs --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/objects_4-16/processed_videos/hs --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135 --video_mode True

# Pokemon Dataset Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_pokemon/augmented --name hyperspectral_pokemon_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_pokemon_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Fruit Dataset Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit/augmented --name hyperspectral_fruit_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Fruit Dataset V2 Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_4-24/augmented/split --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_4-24/test_figure --name hyperspectral_fruit_v2_full --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2_full --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Deep Fruit Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_4-24/augmented/split --name hyperspectral_fruit_v2_deep_wide --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2_deep_wide --model pix2pix --input_nc 1 --output_nc 106 --ngf 128 --netG unet_1024_mod --netG_reps 2 --netD_mult 0 --polarization 0

# Polarized Fruit Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_polarized_fruit_4-24 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_polarized_fruit_4-24 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_polarized_fruit_4-24 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_polarized_fruit_4-24 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Fruit Videos
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_videos/polarized2 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_videos/polarized2 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_videos/polarized2 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_fruit_videos/polarized2 --name hyperspectral_fruit_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_fruit_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135 --video_mode True

# Combined Dataset Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_combined_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_combined_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Combined Testing w/ New Fruit (4/23)
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/fruit_4-23/processed --name hyperspectral_combined_v1 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_combined_v1 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Combined Dataset Testing V2
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/combined_dataset --name hyperspectral_combined_v2 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_combined_v2 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 

# Bug Dataset Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/processed_invertebrates/validation/augmented --name hyperspectral_bugs_v1_8x8 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_bugs_v1_8x8 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Banknote Model Testing - No Polarizer
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-15/no_mask --name hyperspectral_banknotes_b_no_mask --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_b_no_mask --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# Banknote Upsampling Model Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-4/validation/augmented_simple --name hyperspectral_banknotes_G_spatial --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_G_spatial --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0  --GT_upsample True

# Banknote Upsampling Model Training
python train.py \
  --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-4/validation/augmented_simple \
  --name hyperspectral_banknotes_G_spatial \
  --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_G_spatial \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --save_epoch_freq 5 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --epoch latest \
  --polarization 0 \
  --GT_upsample True


# Transfer Learning Res Chart Model
python train.py \
  --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_datasets_testing/spectral_0 \
  --name res_chart_transfer_experiment \
  --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree_transfer \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --save_epoch_freq 5 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --epoch latest \
  --continue_train

# Transfer Learning X-Res Model
python train.py \
  --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_x/spectral/0-90_degree \
  --name res_chart_transfer_experiment \
  --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree_transfer_X \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --save_epoch_freq 5 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --epoch latest \
  --continue_train

# 15 FPS Model Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/15fps_paired --name hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 10 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0
# Color Model Training
python train.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mcbeth_squares --name hyperspectral_3D_SSIM_SPEC_0_degree_mcbeth_squares --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_mcbeth --model pix2pix --input_nc 1 --output_nc 106 --n_epochs 100 --n_epochs_decay 10 --save_epoch_freq 5 --netG unet_1024 --netG_reps 2 --netD_mult 0

# Color Model Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mcbeth_squares --name hyperspectral_3D_SSIM_SPEC_0_degree_mcbeth_squares --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_mcbeth  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0

# Model Testing ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 15 FPS Testing - 0 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/alignment_testing --name hyperspectral_3D_SSIM_SPEC_0_degree_15fps --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_15fps --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
# 15 FPS Testing - 90 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/alignment_testing --name hyperspectral_3D_SSIM_SPEC_15fps_90_degree_test --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_15fps_90_degree_test  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
# 15 FPS Testing - 45 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/alignment_testing  --name hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# OG 0 Degree Testing
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/2-22_0degree --name hyperspectral_1024_large_nogan_double_2-22_0degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0


# Video Generation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 15 FPS - 0 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/slideshow_videos/slideshow_cellophane/slideshow_validation/0_degree --name hyperspectral_3D_SSIM_SPEC_0_degree_15fps --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_0_degree_15fps --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 
# 15 FPS - 90 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/slideshow_videos/slideshow_cellophane/slideshow_validation/90_degree --name hyperspectral_3D_SSIM_SPEC_15fps_90_degree_test --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_15fps_90_degree_test  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
# 15 FPS - 45 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/slideshow_videos/slideshow_cellophane/slideshow_validation/45_degree --name hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_SPEC_15fps_45_degree_test  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135

# Polarized Banknote Video
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/take --name hyperspectral_banknotes_v3 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/take --name hyperspectral_banknotes_v3_45 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_45 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 45 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/take --name hyperspectral_banknotes_v3_90 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_90 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90 --video_mode True
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-14/take --name hyperspectral_banknotes_v3_135 --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_v3_135 --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135 --video_mode True

# Resolution Chart Generation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Transfer Learning
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_datasets_testing/spectral_0 --name res_chart_transfer_experiment --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree_transfer  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
# OG Model
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/banknotes_4-4 --name hyperspectral_1024_large_nogan_double_2-22_0degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# X-Resolution Generation - Transfer Learning
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_x/spectral/0-90_degree --name res_chart_transfer_experiment --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree_transfer_X  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0

# X-Resolution Generation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_x/spectral/0-90_degree --name hyperspectral_1024_large_nogan_double_2-22_0degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-22_0degree  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 0
# 90 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_x/spectral/0-90_degree  --name hyperspectral_1024_large_nogan_double_2-20_90degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-20_90degree --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 90
# 45 Degree
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/resolution_x/spectral/45_degree --name hyperspectral_1024_large_nogan_double_2-24_45degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_2-24_45degree --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0 --polarization 135



# Mcbeth Squares
python test.py --dataroot /scratch/general/nfs1/u1528328/img_dir/mcbeth_squares --name hyperspectral_3D_SSIM_mult_SPEC_0_degree --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_3D_SSIM_mult_SPEC_0_degree  --model pix2pix --input_nc 1 --output_nc 106 --netG unet_1024 --netG_reps 2 --netD_mult 0
