import itertools
import subprocess
import os

# Define the grid of hyperparameters to search
lambda_l1_values = [0.5, 1.0, 2.0]
lambda_gan_values = [0.001, 0.01, 0.1]
lambda_3d_ssim_values = [10.0, 100.0]
lambda_sc_values = [0.5, 1.0]

DATAROOT = "/scratch/general/nfs1/u1528328/img_dir/cumulative_split"
BASE_CHECKPOINTS = "/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperparam_sweep"

# 1. Grid search (manual lambda)
for l1, gan, ssim, sc in itertools.product(lambda_l1_values, lambda_gan_values, lambda_3d_ssim_values, lambda_sc_values):
    run_name = f"sweep_l1{l1}_gan{gan}_ssim{ssim}_sc{sc}"
    checkpoints_dir = os.path.join(BASE_CHECKPOINTS, run_name)
    train_cmd = [
        "python", "train.py",
        "--dataroot", DATAROOT,
        "--name", run_name,
        "--checkpoints_dir", checkpoints_dir,
        "--model", "pix2pix",
        "--input_nc", "1",
        "--output_nc", "106",
        "--n_epochs", "10",
        "--n_epochs_decay", "10",
        "--save_epoch_freq", "5",
        "--netG", "unet_1024",
        "--netG_reps", "2",
        "--netD_mult", "0",
        "--polarization", "0",
        "--dataset_mode", "aligned_augmentation",
        "--lambda_l1", str(l1),
        "--lambda_gan", str(gan),
        "--lambda_3d_ssim", str(ssim),
        "--lambda_sc", str(sc)
    ]
    print("Launching training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

# 2. Auto lambda run (single run)
auto_lambda_name = "sweep_auto_lambda"
auto_lambda_checkpoints = os.path.join(BASE_CHECKPOINTS, auto_lambda_name)
auto_lambda_cmd = [
    "python", "train.py",
    "--dataroot", DATAROOT,
    "--name", auto_lambda_name,
    "--checkpoints_dir", auto_lambda_checkpoints,
    "--model", "pix2pix",
    "--input_nc", "1",
    "--output_nc", "106",
    "--n_epochs", "10",
    "--n_epochs_decay", "10",
    "--save_epoch_freq", "5",
    "--netG", "unet_1024",
    "--netG_reps", "2",
    "--netD_mult", "0",
    "--polarization", "0",
    "--dataset_mode", "aligned_augmentation",
    "--auto_lambda"
]
print("Launching auto_lambda training:", " ".join(auto_lambda_cmd))
subprocess.run(auto_lambda_cmd, check=True)

# 3. Call the evaluation script after all training is done
print("Launching evaluation script...")
subprocess.run(["python", "HSP_test-eval.py"], check=True)