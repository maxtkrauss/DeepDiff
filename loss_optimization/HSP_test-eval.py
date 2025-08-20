import os
import subprocess
import shutil
import csv
import time
from HSI_comparison import HyperspectralAnalyzer

# Updated to match your training script
CHECKPOINTS_ROOT = "/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperparam_old"
DATAROOT = "/scratch/general/nfs1/u1528328/img_dir/cumulative_split"
RESULTS_ROOT = "/scratch/general/nfs1/u1528328/model_dir/W-NET_models/results_hyperparam_old"
CSV_PATH = "hyperparam_old_metrics.csv"

# Filter to only your specific runs
all_dirs = [d for d in os.listdir(CHECKPOINTS_ROOT) if os.path.isdir(os.path.join(CHECKPOINTS_ROOT, d))]
run_names = [d for d in all_dirs if d.startswith("fixed_l1100_gan")]

print(f"Found {len(run_names)} runs to evaluate: {run_names}")

csv_fields = [
    "run_name", "avg_ssim_3d", "avg_ssim_2d", "avg_mse", "avg_mae", "avg_rase", "avg_fidelity", "avg_RSE", "num_images"
]

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()

    for run_name in run_names:
        print(f"\n=== Evaluating run: {run_name} ===")
        checkpoints_dir = os.path.join(CHECKPOINTS_ROOT, run_name)
        result_dir = os.path.join(RESULTS_ROOT, run_name)
        test_cmd = [
            "python", "test.py",
            "--dataroot", DATAROOT,
            "--name", run_name,
            "--checkpoints_dir", checkpoints_dir,
            "--model", "pix2pix",
            "--input_nc", "1",
            "--output_nc", "106",
            "--netG", "unet_1024",
            "--netG_reps", "2",
            "--netD_mult", "0",
            "--polarization", "0",
            "--results_dir", RESULTS_ROOT,
            "--num_test", "161"
        ]
        print("Running test.py...")
        subprocess.run(test_cmd, check=True)

        # ...rest of your evaluation code...

        # 2. Evaluate metrics using HSI_comparison.py logic
        images_dir = os.path.join(result_dir, "test_latest", "images")
        if not os.path.exists(images_dir):
            print(f"Images directory not found for run {run_name}, skipping.")
            continue

        try:
            analyzer = HyperspectralAnalyzer(images_dir)
            metrics = analyzer.calculate_all_metrics()
        except Exception as e:
            print(f"Error evaluating metrics for run {run_name}: {e}")
            continue

        # 3. Write metrics to CSV
        metrics_row = {"run_name": run_name}
        metrics_row.update(metrics)
        writer.writerow(metrics_row)
        csvfile.flush()
        print(f"Metrics for {run_name} written to CSV.")

        # 4. Delete generated images to save space
        print(f"Deleting generated images for {run_name}...")
        try:
            shutil.rmtree(os.path.join(result_dir, "test_latest"))
        except Exception as e:
            print(f"Error deleting images for {run_name}: {e}")

        # Optional: sleep to avoid overloading filesystem
        time.sleep(2)

print("All runs evaluated and metrics saved to", CSV_PATH)