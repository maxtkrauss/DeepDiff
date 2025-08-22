
import os
import subprocess
import shutil
import pandas as pd
import argparse

# Define datasets and model/checkpoint directories

# Example dataset definition matching banknotes_training.sh
DATASETS = [
    {
        'name': 'banknotes_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_banknotes_augmented',
    },
    {
        'name': 'william_summer_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/William_Summer/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_william_summer_augmented',
    },
    {
        'name': 'produce_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/produce/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_produce_augmented',
    },
    {
        'name': 'invertebrates_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/invertebrates/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_invertebrates_augmented',
    },
    {
        'name': 'rescharts_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/rescharts/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_rescharts_augmented',
    },
    {
        'name': 'currency_dataset_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/currency_dataset/augmented',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_currency_dataset_augmented',
    },
]

POL_ANGLES = [0, 45, 90, 135]

TRAIN_SCRIPT = 'train.py'
TEST_SCRIPT = 'test.py'
EVAL_SCRIPT = 'HSI_comparison.py'
RESULTS_DIR = 'results'  # Directory where test images are saved



# Fixed options for training and testing, matching banknotes_training.sh
TRAIN_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '106',
    '--n_epochs', '10',
    '--n_epochs_decay', '10',
    '--save_epoch_freq', '5',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise'
]
TEST_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '106',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise'
]

# Helper to run a command and print output
def run_cmd(cmd):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to process (from DATASETS)')
    args = parser.parse_args()

    selected_datasets = DATASETS
    if args.dataset:
        selected_datasets = [ds for ds in DATASETS if ds['name'] == args.dataset]
        if not selected_datasets:
            raise ValueError(f"Dataset {args.dataset} not found in DATASETS.")

    all_metrics = []
    for ds in selected_datasets:
        dataset_name = ds['name']
        for pol in POL_ANGLES:
            model_name = ds['name']
            ckpt_dir = ds['checkpoints_dir']
            # 1. Train
            train_cmd = [
                'python', TRAIN_SCRIPT,
                '--dataroot', ds['dataroot'],
                '--name', model_name,
                '--checkpoints_dir', ckpt_dir,
                '--polarization', str(pol),
            ] + TRAIN_OPTS
            run_cmd(train_cmd)
            # 2. Test
            test_cmd = [
                'python', TEST_SCRIPT,
                '--dataroot', ds['dataroot'],
                '--name', model_name,
                '--checkpoints_dir', ckpt_dir,
                '--polarization', str(pol),
            ] + TEST_OPTS
            run_cmd(test_cmd)
            # 3. Evaluate
            eval_img_dir = os.path.join(RESULTS_DIR, model_name, 'validation_latest', 'images')
            eval_cmd = [
                'python', EVAL_SCRIPT,
                '--results_dir', eval_img_dir
            ]
            run_cmd(eval_cmd)
            # 4. Read metrics and append to master CSV
            metrics_csv = os.path.join(eval_img_dir, 'metrics.csv')
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                df['dataset'] = ds['name']
                df['polarization'] = pol
                all_metrics.append(df)
            else:
                print(f"Warning: Metrics file not found for {model_name} pol {pol}")
            # 5. Delete test images to save space
            test_img_dir = os.path.join(RESULTS_DIR, model_name)
            if os.path.exists(test_img_dir):
                shutil.rmtree(test_img_dir)
                print(f"Deleted test images in {test_img_dir}")
    # Save all metrics to a unique master CSV per dataset
    if all_metrics:
        master_csv_name = f"master_metrics_{dataset_name}.csv"
        master_df = pd.concat(all_metrics, ignore_index=True)
        master_df.to_csv(master_csv_name, index=False)
        print(f"Master metrics written to {master_csv_name}")
    else:
        print("No metrics collected.")

if __name__ == "__main__":
    main()
