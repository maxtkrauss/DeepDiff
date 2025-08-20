import subprocess
import os

# Set the fixed hyperparameters
lambda_l1 = 100
lambda_3d_ssim = 100
lambda_sc = 1.0

DATAROOT = "/scratch/general/nfs1/u1528328/img_dir/cumulative_split"
BASE_CHECKPOINTS = "/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperparam_old"

# List of GAN values to try
gan_values = [0, 0.01]

for lambda_gan in gan_values:
    # run_name = f"fixed_l1{lambda_l1}_gan{lambda_gan}_ssim{lambda_3d_ssim}_sc{lambda_sc}"
    # checkpoints_dir = os.path.join(BASE_CHECKPOINTS, run_name)
    # train_cmd = [
    #     "python", "train.py",
    #     "--dataroot", DATAROOT,
    #     "--name", run_name,
    #     "--checkpoints_dir", checkpoints_dir,
    #     "--model", "pix2pix",
    #     "--input_nc", "1",
    #     "--output_nc", "106",
    #     "--n_epochs", "10",
    #     "--n_epochs_decay", "10",
    #     "--save_epoch_freq", "5",
    #     "--netG", "unet_1024",
    #     "--netG_reps", "2",
    #     "--netD_mult", "0",
    #     "--polarization", "0",
    #     "--dataset_mode", "aligned_augmentation",
    #     "--lambda_l1", str(lambda_l1),
    #     "--lambda_gan", str(lambda_gan),
    #     "--lambda_3d_ssim", str(lambda_3d_ssim),
    #     "--lambda_sc", str(lambda_sc)
    # ]
    # print("Launching training:", " ".join(train_cmd))
    # subprocess.run(train_cmd, check=True)

    # Call the evaluation script after training is done
    print("Launching evaluation script...")
    subprocess.run(["python", "HSP_test-eval.py"], check=True)
