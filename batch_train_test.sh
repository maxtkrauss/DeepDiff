#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=training_job_$1
#SBATCH --output=training_job_%j.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1528328@utah.edu

## Activate the conda environment
source activate hsp_env

## Navigate to the project directory
cd /uufs/chpc.utah.edu/common/home/u1528328/DiffuserNET

python batch_train_test.py --dataset $1

sbatch batch_train_test.sh banknotes_augmented
sbatch batch_train_test.sh william_summer_augmented
sbatch batch_train_test.sh produce_augmented
sbatch batch_train_test.sh invertebrates_augmented
sbatch batch_train_test.sh rescharts_augmented
sbatch batch_train_test.sh currency_dataset_augmented