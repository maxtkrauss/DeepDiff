import os
import subprocess
import shutil
import csv
import time
import argparse
from HSI_comparison import HyperspectralAnalyzer

def evaluate_training_runs(checkpoints_root, results_root, csv_path, dataroot, run_filter=None, test_polarization="0"):
    """
    Flexible evaluation script that works with any training configuration
    """
    # Get all checkpoint directories
    if not os.path.exists(checkpoints_root):
        print(f"Checkpoints directory not found: {checkpoints_root}")
        return
    
    all_dirs = [d for d in os.listdir(checkpoints_root) if os.path.isdir(os.path.join(checkpoints_root, d))]
    
    # Apply filter if provided
    if run_filter:
        run_names = [d for d in all_dirs if run_filter in d]
    else:
        run_names = all_dirs
    
    print(f"Found {len(run_names)} runs to evaluate: {run_names}")
    
    if not run_names:
        print("No runs found matching criteria!")
        return
    
    # Create results directory
    os.makedirs(results_root, exist_ok=True)
    
    # CSV fields for metrics
    csv_fields = [
        "run_name", "avg_ssim_3d", "avg_ssim_2d", "avg_mse", "avg_mae", 
        "avg_rase", "avg_fidelity", "avg_RSE", "num_images"
    ]
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        
        for run_name in run_names:
            print(f"\n=== Evaluating run: {run_name} ===")
            
            # Extract polarization from run name if it contains polarization info
            run_polarization = test_polarization
            if "pol_" in run_name:
                try:
                    # Extract polarization from run name like "pol_45deg_..."
                    pol_part = run_name.split("pol_")[1].split("deg")[0]
                    run_polarization = pol_part
                    print(f"Detected polarization from run name: {run_polarization}")
                except:
                    print(f"Could not extract polarization from run name, using default: {test_polarization}")
            
            checkpoints_dir = os.path.join(checkpoints_root, run_name)
            result_dir = os.path.join(results_root, run_name)
            
            # Build test command
            test_cmd = [
                "python", "test.py",
                "--dataroot", dataroot,
                "--name", run_name,
                "--checkpoints_dir", checkpoints_dir,
                "--model", "pix2pix",
                "--input_nc", "1",
                "--output_nc", "106",
                "--netG", "unet_1024",
                "--netG_reps", "2",
                "--netD_mult", "0",
                "--polarization", run_polarization,
                "--results_dir", results_root,
                "--num_test", "161"
            ]
            
            print("Running test.py...")
            print("Command:", " ".join(test_cmd))
            
            try:
                subprocess.run(test_cmd, check=True)
                print(f"Testing completed for {run_name}")
            except subprocess.CalledProcessError as e:
                print(f"Testing failed for {run_name}: {e}")
                continue
            
            # Evaluate metrics using HSI_comparison.py logic
            images_dir = os.path.join(result_dir, "test_latest", "images")
            if not os.path.exists(images_dir):
                print(f"Images directory not found for run {run_name}, skipping.")
                continue
            
            try:
                print("Calculating metrics...")
                analyzer = HyperspectralAnalyzer(images_dir)
                metrics = analyzer.calculate_all_metrics()
                print(f"Metrics calculated for {run_name}")
            except Exception as e:
                print(f"Error evaluating metrics for run {run_name}: {e}")
                continue
            
            # Write metrics to CSV
            metrics_row = {"run_name": run_name}
            metrics_row.update(metrics)
            writer.writerow(metrics_row)
            csvfile.flush()
            print(f"Metrics for {run_name} written to CSV.")
            
            # Delete generated images to save space
            print(f"Deleting generated images for {run_name}...")
            try:
                shutil.rmtree(os.path.join(result_dir, "test_latest"))
            except Exception as e:
                print(f"Error deleting images for {run_name}: {e}")
            
            # Sleep to avoid overloading filesystem
            time.sleep(2)
    
    print(f"\nAll runs evaluated and metrics saved to: {csv_path}")

# Dynamic evaluation based on available checkpoint directories
def evaluate_all_polarization_experiments():
    """Automatically detect and evaluate all polarization experiments"""
    base_path = "/scratch/general/nfs1/u1528328/model_dir/W-NET_models/"
    
    # Look for all polarization checkpoint directories
    polarization_dirs = [
        "checkpoints_polarization_cumulative",
        "checkpoints_polarization_currency"
    ]
    
    for checkpoint_dir in polarization_dirs:
        full_checkpoint_path = os.path.join(base_path, checkpoint_dir)
        
        if os.path.exists(full_checkpoint_path):
            print(f"\nEvaluating experiments in: {checkpoint_dir}")
            
            # Determine dataset and results path
            if "cumulative" in checkpoint_dir:
                dataroot = "/scratch/general/nfs1/u1528328/img_dir/cumulative_split"
                results_root = os.path.join(base_path, "results_polarization_cumulative")
                csv_path = "polarization_cumulative_metrics.csv"
            elif "currency" in checkpoint_dir:
                dataroot = "/scratch/general/nfs1/u1528328/img_dir/currency_dataset"
                results_root = os.path.join(base_path, "results_polarization_currency")
                csv_path = "polarization_currency_metrics.csv"
            else:
                # Default fallback
                dataroot = "/scratch/general/nfs1/u1528328/img_dir/cumulative_split"
                results_root = os.path.join(base_path, f"results_{checkpoint_dir}")
                csv_path = f"{checkpoint_dir}_metrics.csv"
            
            evaluate_training_runs(
                checkpoints_root=full_checkpoint_path,
                results_root=results_root,
                csv_path=csv_path,
                dataroot=dataroot,
                run_filter="pol_",
                test_polarization="0"
            )
        else:
            print(f"Checkpoint directory not found: {full_checkpoint_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description="Flexible HSP model evaluation script")
        parser.add_argument("--checkpoints_root", required=True, help="Directory containing model checkpoints")
        parser.add_argument("--results_root", required=True, help="Directory to save test results")
        parser.add_argument("--csv_path", required=True, help="Path to save metrics CSV file")
        parser.add_argument("--dataroot", required=True, help="Path to test dataset")
        parser.add_argument("--run_filter", default=None, help="Filter for run names")
        parser.add_argument("--test_polarization", default="0", help="Polarization to use during testing")
        
        args = parser.parse_args()
        evaluate_training_runs(
            checkpoints_root=args.checkpoints_root,
            results_root=args.results_root,
            csv_path=args.csv_path,
            dataroot=args.dataroot,
            run_filter=args.run_filter,
            test_polarization=args.test_polarization
        )
    else:
        # Automatic mode - evaluate all polarization experiments
        print("Auto-detecting polarization experiments...")
        evaluate_all_polarization_experiments()