
#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=training_job_banknotes_8-21
#SBATCH --output=training_job_banknotes_8-21_%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1528328@utah.edu

################################################################################
# Example SLURM script for training and testing a Pix2Pix model on banknotes data
#
# This script demonstrates how to train and test a single model using the
# DiffuserNET codebase. It is intended as a reference for new users and as a
# reproducible portkey for running and evaluating models in this directory.
#
# Key points:
# - Training and testing are performed using train.py and test.py.
# - Data is read from the /scratch (fast, temporary) directory for performance.
# - Model checkpoints are saved to /scratch for speed and to avoid quota issues.
# - The project code and scripts are stored in your /uufs (Z:) home directory.
# - Generated images are written to the results/ folder in the project directory.
# - After testing, you can use HSI_comparison.py to evaluate model performance.
#
# Directory structure (for this example):
#   /uufs/chpc.utah.edu/common/home/u1528328/DiffuserNET/   # Project code (Z:)
#   /scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes # Training/test data
#   /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_8-21 # Model checkpoints
#   results/banknotes_augmented/validation_latest/images/    # Generated images
#
# /uufs (Z:) is your persistent home directory (safe for code, scripts, results).
# /scratch is a fast, temporary storage area for large data and model files.
#
# Training and testing workflow:
# 1. Activate the conda environment with all dependencies.
# 2. Change to the project directory (on /uufs/Z:).
# 3. Run train.py to train the model. Checkpoints are saved to /scratch.
# 4. Run test.py to generate output images. Images are written to results/.
# 5. Use HSI_comparison.py to evaluate the generated images and write metrics.
# 6. All results and metrics can be found in the results/ directory.
#
# Example usage:
#   sbatch banknotes_training.sh
#
# After completion, see results in:
#   results/banknotes_augmented/validation_latest/images/
#   results/banknotes_augmented/validation_latest/metrics.csv (after evaluation)
#
# To evaluate model performance:
#   python HSI_comparison.py --results_dir results/banknotes_augmented/validation_latest/images
#
################################################################################

## Activate the conda environment
source activate hsp_env

## Navigate to the project directory (on /uufs/Z:)
cd /uufs/chpc.utah.edu/common/home/u1528328/DiffuserNET

# Train the model (checkpoints saved to /scratch)
python train.py \
    --dataroot /scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes \
    --name hyperspectral_banknotes_8-21 \
    --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_8-21 \
    --model pix2pix \
    --input_nc 1 \
    --output_nc 106 \
    --n_epochs 10 \
    --n_epochs_decay 10 \
    --save_epoch_freq 5 \
    --netG unet_1024 \
    --netG_reps 2 \
    --netD_mult 0 \
    --polarization 0 \
    --norm_bitwise

# Test the model (generated images written to results/)
python test.py \
    --dataroot /scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes \
    --name hyperspectral_banknotes_8-21 \
    --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_banknotes_8-21 \
    --model pix2pix \
    --input_nc 1 \
    --output_nc 106 \
    --netG unet_1024 \
    --netG_reps 2 \
    --netD_mult 0 \
    --polarization 0 \
    --norm_bitwise

# Evaluate the model (optional, after test.py)
python HSI_comparison.py --results_dir results/banknotes_augmented/validation_latest/images