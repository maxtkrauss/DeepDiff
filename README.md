# Hyperspectral Image-to-Image Translation

This repository provides a general-purpose framework for training and testing image-to-image translation models, with a focus on hyperspectral image (HSI) data. The codebase is adapted from the popular [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project, with customizations for hyperspectral data and additional evaluation tools.

## Features
- **Supports multiple models:** CycleGAN, pix2pix, colorization, and custom models.
- **Flexible dataset handling:** Aligned, unaligned, single, and colorization dataset modes.
- **Training and testing scripts:** Modular scripts for training (`train.py`) and testing (`test.py`) models.
- **Visualization:** HTML and Visdom-based visualization of results and loss curves.
- **Metrics:** Includes tools for evaluating SSIM, MSE, MAE, RASE, spectral fidelity, and more (see `HSI_comparison.py`).
- **Extensible:** Easily add new models, datasets, and evaluation metrics.

## Directory Structure
```
HSP/
├── train.py                # Main training script
├── test.py                 # Main testing script
├── HSI_comparison.py       # Hyperspectral evaluation and visualization
├── data/                   # Dataset classes and loaders
├── models/                 # Model architectures and networks
├── options/                # Command-line options and argument parsing
├── util/                   # Utility functions and visualization tools
├── results/                # Output directory for results and HTML visualizations
└── ...
```

## Installation
1. **Clone the repository:**
   ```sh
   git clone <this-repo-url>
   cd HSP
   ```
2. **Install dependencies:**
   - Python 3.7+
   - PyTorch (see [PyTorch installation guide](https://pytorch.org/get-started/locally/))
   - numpy, scikit-image, matplotlib, tifffile, wandb (optional)
   - Install all requirements:
     ```sh
     pip install -r requirements.txt
     ```
   - If `requirements.txt` is missing, install manually:
     ```sh
     pip install torch numpy scikit-image matplotlib tifffile wandb
     ```

## Usage

### 1. Training
Train a model using the `train.py` script. Example:

```sh
python train.py \
  --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 \
  --name Geology_Sample_1_pol0_v1 \
  --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol0_v1 \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --save_epoch_freq 5 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --polarization 0
```

**Key arguments:**
- `--dataroot`: Path to the dataset root directory.
- `--name`: Name of the experiment (used for saving checkpoints and results).
- `--checkpoints_dir`: Directory to save/load model checkpoints.
- `--model`: Model type (`cycle_gan`, `pix2pix`, `colorization`, etc.).
- `--input_nc`: Number of input channels (e.g., 1 for grayscale, 3 for RGB, or 1 for single-band HSI input).
- `--output_nc`: Number of output channels (e.g., 106 for hyperspectral output).
- `--n_epochs`: Number of epochs with initial learning rate.
- `--n_epochs_decay`: Number of epochs to linearly decay learning rate to zero.
- `--save_epoch_freq`: Frequency (in epochs) to save checkpoints.
- `--netG`: Generator architecture (e.g., `unet_1024`).
- `--netG_reps`: Number of generator repetitions (custom argument).
- `--netD_mult`: Discriminator multiplier (custom argument).
- `--polarization`: Polarization flag (custom argument).

See `options/base_options.py` and `options/train_options.py` for all available options.

### 2. Testing
Test a trained model using the `test.py` script. Example:

```sh
python test.py \
  --dataroot /scratch/general/nfs1/u1528328/img_dir/wwalker/Geology_Sample_1 \
  --name Geology_Sample_1_pol0_v1 \
  --checkpoints_dir /scratch/general/nfs1/u1528328/model_dir/W-NET_models/checkpoints_hyperspectral_Geology_Sample_1_pol0_v1 \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --polarization 0
```

**Key arguments:**
- `--dataroot`: Path to the dataset root directory.
- `--name`: Name of the experiment (should match the training run).
- `--checkpoints_dir`: Directory to load model checkpoints from.
- `--model`: Model type (`cycle_gan`, `pix2pix`, `test`, etc.).
- `--input_nc`: Number of input channels.
- `--output_nc`: Number of output channels.
- `--netG`: Generator architecture.
- `--netG_reps`: Number of generator repetitions.
- `--netD_mult`: Discriminator multiplier.
- `--polarization`: Polarization flag.
- `--results_dir`: Directory to save test results (default: `./results`).
- `--epoch`: Which epoch to test (default: `latest`).
- `--num_test`: Number of test images to process.

Test results (images and HTML visualizations) are saved in the `results/` directory.

### 3. Hyperspectral Evaluation and Visualization
The `HSI_comparison.py` script provides tools for:
- Loading and comparing hyperspectral ground truth and reconstructed images.
- Calculating metrics: SSIM (2D/3D), MSE, MAE, RASE, spectral fidelity, and more.
- Interactive visualization of spectral curves and RGB renderings.

**Usage:**
Edit the `image_directory` and `NUM_IMAGES` variables in `HSI_comparison.py` to point to your results, then run:
```sh
python HSI_comparison.py
```

### 4. Customization
- **Add new datasets:** Implement a new class in `data/` and register it in `data/__init__.py`.
- **Add new models:** Implement a new class in `models/` and register it in `models/__init__.py`.
- **Add new metrics:** Extend `HSI_comparison.py` or add new utility functions in `util/`.

## Checkpoints and Results
- Checkpoints are saved in `./checkpoints/<experiment_name>/`.
- Test results and HTML visualizations are saved in `./results/<experiment_name>/<phase>_<epoch>/`.

## Tips and References
- For more details on options and tips, see the original [CycleGAN and pix2pix documentation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- For hyperspectral-specific evaluation, see the comments and docstrings in `HSI_comparison.py`.

## Acknowledgements
- This codebase is based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- Additional hyperspectral evaluation tools and metrics by project contributors.

## License
See `LICENSE` file for details.
