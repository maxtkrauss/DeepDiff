import subprocess
import os

def train_and_evaluate(DATAROOT, BASE_CHECKPOINTS, polarizations, lambda_l1, lambda_3d_ssim, lambda_sc, lambda_gan):
    print("Training models for each polarization angle...")
    print("="*60)
    for polarization in polarizations:
        for use_norm_bitwise in [False, True]:
            norm_bitwise_flag = "--norm_bitwise" if use_norm_bitwise else ""
            run_name = f"pol_{polarization}deg_l1{lambda_l1}_gan{lambda_gan}_ssim{lambda_3d_ssim}_sc{lambda_sc}"
            if use_norm_bitwise:
                run_name += "_normbitwise"
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
                "--polarization", str(polarization),
                "--dataset_mode", "aligned_augmentation",
                "--lambda_l1", str(lambda_l1),
                "--lambda_gan", str(lambda_gan),
                "--lambda_3d_ssim", str(lambda_3d_ssim),
                "--lambda_sc", str(lambda_sc)
            ]
            if use_norm_bitwise:
                train_cmd.append("--norm_bitwise")
            print(f"Training model for {polarization}° polarization... (norm_bitwise={use_norm_bitwise})")
            print(f"Run name: {run_name}")
            print("Command:", " ".join(train_cmd))
            try:
                subprocess.run(train_cmd, check=True)
                print(f"Training completed for {polarization}° polarization (norm_bitwise={use_norm_bitwise})\n")
            except subprocess.CalledProcessError as e:
                print(f"Training failed for {polarization}° polarization (norm_bitwise={use_norm_bitwise}): {e}")
                continue

    print("All training completed! Launching evaluation script...")
    print("="*60)
    try:
        subprocess.run(["python", "flexible_HSP_test-eval.py"], check=True)
        print("All evaluations completed!")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")

    print("Polarization training experiment complete!")
    print("Check polarization_metrics.csv to see which polarization angle performs best.")

# Fixed hyperparameters
lambda_l1 = 12.0
lambda_3d_ssim = 100
lambda_sc = 0.5
lambda_gan = 0.003
polarizations = [0, 45, 90, 135]

# First dataset
train_and_evaluate(
    DATAROOT="/scratch/general/nfs1/u1528328/img_dir/cumulative_split",
    BASE_CHECKPOINTS="/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_polarization_cumulative",
    polarizations=polarizations,
    lambda_l1=lambda_l1,
    lambda_3d_ssim=lambda_3d_ssim,
    lambda_sc=lambda_sc,
    lambda_gan=lambda_gan
)

# Second dataset
train_and_evaluate(
    DATAROOT="/scratch/general/nfs1/u1528328/img_dir/currency_dataset",
    BASE_CHECKPOINTS="/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_polarization_currency",
    polarizations=polarizations,
    lambda_l1=lambda_l1,
    lambda_3d_ssim=lambda_3d_ssim,
    lambda_sc=lambda_sc,
    lambda_gan=lambda_gan
)
