# Hyperspectral Image-to-Image Translation

This repository provides a comprehensive framework for training and testing image-to-image translation models, with a focus on hyperspectral image (HSI) and polarimetric data. The codebase is adapted from the popular [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project, with extensive customizations for hyperspectral/polarimetric data, advanced preprocessing tools, and specialized evaluation capabilities.

## Features
- **Supports multiple models:** CycleGAN, pix2pix, colorization, and custom models with attention mechanisms.
- **Flexible dataset handling:** Aligned, unaligned, single, colorization, and 16-bit dataset modes.
- **Advanced preprocessing:** Multi-modal image alignment, augmentation, and polarimetric video analysis.
- **Training and testing scripts:** Modular scripts for training (`train.py`) and testing (`test.py`) with flexible evaluation.
- **Hyperparameter optimization:** Automated hyperparameter sweeps and analysis tools.
- **Polarimetric analysis:** Complete DoLP/AoLP video generation and Stokes parameter calculation.
- **Visualization:** HTML and Visdom-based visualization, spectral plotting, and interactive image viewers.
- **Comprehensive metrics:** SSIM (2D/3D), MSE, MAE, RASE, spectral fidelity, correlation coefficients, and more.
- **Extensible:** Easily add new models, datasets, alignment methods, and evaluation metrics.

## Directory Structure
```
HSP/
├── train.py                      # Main training script
├── test.py                       # Main testing script  
├── flexible_HSP_test-eval.py     # Flexible evaluation script for any dataset/model
├── polarimetric_training.py      # Multi-dataset polarimetric training script
├── HSI_comparison.py             # Hyperspectral evaluation and visualization
├── data/                         # Dataset classes and loaders
│   ├── aligned_dataset.py        # Standard aligned dataset
│   ├── aligned_augmentation_dataset.py  # Dataset with augmentation
│   ├── 16-bit_dataloader/        # 16-bit data handling
│   └── cave_dataloaders/         # CAVE dataset loaders
├── models/                       # Model architectures and networks
│   ├── pix2pix_model.py          # Pix2Pix implementation
│   ├── cycle_gan_model.py        # CycleGAN implementation
│   ├── networks.py               # Neural network architectures
│   ├── networks_attention.py     # Attention mechanism networks
│   └── networks_cave.py          # CAVE-specific networks
├── options/                      # Command-line options and argument parsing
├── util/                         # Utility functions and visualization tools
├── image_preprocessing/          # Image alignment and preprocessing tools
│   ├── img_alignment.py          # General image alignment
│   ├── img_alignment_thorlabs.py # Thorlabs-specific cropping tool
│   ├── img_augmentation.py       # Data augmentation
│   └── alignment_testing/        # Advanced alignment algorithms
│       ├── panchromatic_alignment.py     # Main panchromatic alignment tool
│       └── template_matching_alignment.py # Template matching approach
├── video_preprocessing/          # Video and polarimetric analysis
│   └── thorlabs_video_testing.py # DoLP/AoLP video generation
├── loss_optimization/            # Hyperparameter optimization tools
│   ├── hyperparam_finder.py      # Automated hyperparameter sweeps
│   ├── HSP_test-eval.py          # Original evaluation script
│   └── analyze_hyperparam_results.py # Hyperparameter analysis
├── image_visualization/          # Visualization and plotting tools
│   ├── image_viewer.py           # Interactive image viewer
│   └── wavelength_plotting.py    # Spectral plotting utilities
├── training_calls/              # Training configuration scripts
├── RGB/                         # RGB conversion and color matching
├── results/                     # Output directory for results and HTML visualizations
└── requirements.txt            # Python dependencies
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
   - OpenCV, numpy, scikit-image, matplotlib, tifffile, pandas, wandb (optional)
   - Install all requirements:
     ```sh
     pip install -r requirements.txt
     ```
   - Key packages include:
     ```sh
     pip install torch opencv-python numpy scikit-image matplotlib tifffile pandas wandb
     ```

## Quick Start

### Image Alignment and Preprocessing
Before training, align your multichannel images using the preprocessing tools:

**For general alignment (template matching with panchromatic averaging):**
```sh
cd image_preprocessing/alignment_testing
python panchromatic_alignment.py
```

**For Thorlabs-specific cropping:**
```sh
cd image_preprocessing  
python img_alignment_thorlabs.py
```

**For data augmentation:**
```sh
cd image_preprocessing
python img_augmentation.py
```

## Usage

### 1. Standard Training
Train a model using the `train.py` script. Example:

```sh
python train.py \
  --dataroot /path/to/your/dataset \
  --name experiment_name \
  --checkpoints_dir /path/to/checkpoints \
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

### 2. Multi-Dataset Polarimetric Training  
For training across multiple polarization angles, use the automated script:

```sh
python polarimetric_training.py
```

This script automatically:
- Trains models for polarization angles 0°, 45°, 90°, 135°
- Handles multiple datasets in sequence
- Automatically runs evaluation after each training
- Saves comprehensive metrics and visualizations

### 3. Flexible Evaluation
Use the flexible evaluation script for any trained model:

```sh
python flexible_HSP_test-eval.py \
  --base_dir /path/to/results \
  --model_name your_model_name \
  --polarization_angles 0 45 90 135
```

**Key features:**
- Automatically detects available datasets and models
- Calculates comprehensive metrics (SSIM, MSE, MAE, RASE, spectral correlation)
- Saves detailed CSV reports with per-image and aggregate statistics
- Handles both single and multi-polarization evaluations

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

### 4. Testing Trained Models
Test a trained model using the `test.py` script:

```sh
python test.py \
  --dataroot /path/to/your/dataset \
  --name experiment_name \
  --checkpoints_dir /path/to/checkpoints \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 106 \
  --netG unet_1024 \
  --netG_reps 2 \
  --netD_mult 0 \
  --polarization 0
```

### 5. Polarimetric Video Analysis
Generate DoLP (Degree of Linear Polarization) and AoLP (Angle of Linear Polarization) videos from polarimetric image sequences:

```sh
cd video_preprocessing
python thorlabs_video_testing.py
```

**Features:**
- Processes polarimetric TIFF sequences (0°, 45°, 90°, 135°)
- Calculates Stokes parameters and polarization metrics
- Generates enhanced DoLP/AoLP videos with colorbars
- Applies contrast enhancement and sharpening
- Outputs at 35 FPS with proper frame timing

### 6. Hyperparameter Optimization
Find optimal training parameters using automated sweeps:

**Run hyperparameter sweep:**
```sh  
cd loss_optimization
python hyperparam_finder.py
```

**Analyze results:**
```sh
python analyze_hyperparam_results.py
```

The analysis script provides:
- Composite scoring based on multiple metrics
- Best parameter identification
- Statistical summaries and rankings
- CSV output for further analysis

### 7. Advanced Image Alignment

**Panchromatic Alignment (Recommended):**
- Uses averaged panchromatic images for template matching
- Handles different imaging modalities (Thorlabs vs Cubert)
- Preserves native resolution while enabling alignment
```sh
cd image_preprocessing/alignment_testing
python panchromatic_alignment.py
```

**Template Matching Alignment:**
- Alternative approach using highest variance channels  
- CLAHE enhancement for low-contrast images
```sh  
cd image_preprocessing/alignment_testing
python template_matching_alignment.py
```

## Command Line Arguments

### Core Training/Testing Arguments
- `--dataroot`: Path to the dataset root directory
- `--name`: Name of the experiment (used for saving checkpoints and results)
- `--checkpoints_dir`: Directory to save/load model checkpoints
- `--model`: Model type (`cycle_gan`, `pix2pix`, `colorization`, `test`)
- `--input_nc`: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
- `--output_nc`: Number of output channels (e.g., 106 for hyperspectral output)
- `--netG`: Generator architecture (`unet_1024`, `unet_512`, `resnet_9blocks`)
- `--netG_reps`: Number of generator repetitions (custom parameter)
- `--netD_mult`: Discriminator multiplier (custom parameter)
- `--polarization`: Polarization angle flag (0, 45, 90, 135)

### Training-Specific Arguments  
- `--n_epochs`: Number of epochs with initial learning rate
- `--n_epochs_decay`: Number of epochs to linearly decay learning rate to zero
- `--save_epoch_freq`: Frequency (in epochs) to save checkpoints
- `--lr`: Initial learning rate (default: 0.0002)
- `--beta1`: Beta1 parameter for Adam optimizer (default: 0.5)
- `--lambda_L1`: Weight for L1 loss (default: 100.0)

### Testing-Specific Arguments
- `--results_dir`: Directory to save test results (default: `./results`)
- `--epoch`: Which epoch to test (default: `latest`)
- `--num_test`: Number of test images to process
- `--phase`: Dataset phase to test (`test`, `val`, `train`)

### Flexible Evaluation Arguments
- `--base_dir`: Base directory containing results
- `--model_name`: Name pattern for models to evaluate
- `--polarization_angles`: List of polarization angles to process
- `--output_dir`: Directory for evaluation outputs

See `options/base_options.py`, `options/train_options.py`, and `options/test_options.py` for complete argument lists.

### 8. Hyperspectral Evaluation and Visualization
The `HSI_comparison.py` script provides comprehensive evaluation tools:

**Usage:**
```sh
python HSI_comparison.py
```

**Features:**
- Loading and comparing hyperspectral ground truth and reconstructed images
- Comprehensive metrics: SSIM (2D/3D), MSE, MAE, RASE, spectral fidelity
- Correlation coefficients and spectral angle mapper (SAM)
- Interactive visualization of spectral curves and RGB renderings
- Batch processing capabilities

**Interactive Image Viewer:**
```sh
cd image_visualization
python image_viewer.py
```

### 9. Data Preprocessing and Augmentation

**Train/Test Split:**
```sh
cd image_preprocessing  
python train_test_split.py
```

**Image Shape Analysis:**
```sh
cd image_preprocessing
python img_shapes.py  
```

**Advanced Augmentation:**
```sh
cd image_preprocessing
python img_augmentation_WILL.py  # William's augmentation methods
```

## Advanced Features

### Multi-Modal Image Alignment
The framework includes sophisticated alignment tools for handling images from different acquisition systems:

**Panchromatic Alignment Approach:**
- Creates panchromatic (averaged) reference images from multichannel data
- Uses template matching for robust alignment across different modalities
- Handles resolution differences between imaging systems (e.g., Thorlabs 660x660 vs Cubert 120x120)
- Preserves native resolution while enabling accurate spatial registration

**CLAHE Enhancement:**
- Contrast Limited Adaptive Histogram Equalization for low-contrast images
- Improves alignment accuracy for challenging imaging conditions
- Configurable clip limits and tile grid sizes

### 16-Bit Data Support
Full support for 16-bit TIFF data with specialized loaders:
- Custom dataset classes for high bit-depth imagery
- Proper normalization and scaling for deep learning models
- Memory-efficient loading for large hyperspectral cubes

### Attention Mechanisms
Advanced neural network architectures with attention:
- Self-attention modules for improved feature learning
- Spatial and spectral attention for hyperspectral data
- Custom attention implementations in `networks_attention.py`

### CAVE Dataset Integration
Specialized support for the CAVE multispectral database:
- Custom dataloaders optimized for CAVE format
- Preprocessing pipelines for CAVE-specific workflows
- Network architectures tuned for CAVE characteristics

### Polarimetric Analysis Pipeline
Complete polarimetric imaging analysis:
- Stokes parameter calculation (I, Q, U, V)
- DoLP and AoLP computation with proper normalization
- Video generation with synchronized colorbars
- Contrast enhancement for low-polarization samples
- Frame-by-frame processing with temporal consistency

### Hyperparameter Optimization
Automated hyperparameter search and analysis:
- Grid search and random search capabilities
- Multi-metric optimization with composite scoring
- Statistical analysis and parameter ranking
- CSV export for external analysis tools

## Customization and Extension

### Adding New Models
1. Create a new model class in `models/your_model.py`
2. Inherit from `BaseModel` and implement required methods
3. Register the model in `models/__init__.py`
4. Add model-specific options in `options/`

### Adding New Datasets
1. Create a new dataset class in `data/your_dataset.py`
2. Inherit from `BaseDataset` and implement required methods
3. Register the dataset in `data/__init__.py`
4. Handle dataset-specific preprocessing as needed

### Adding New Networks
1. Implement new architectures in `models/networks.py`
2. For specialized networks, create separate files (e.g., `networks_attention.py`)
3. Add network selection logic in model classes
4. Document new network parameters in options

### Adding New Metrics
1. Extend `HSI_comparison.py` with new evaluation functions
2. Add metric computation to flexible evaluation scripts
3. Update CSV output formats to include new metrics
4. Consider visualization needs for new metrics

## Workflows and Best Practices

### Complete Training Pipeline
1. **Data Preparation:**
   ```sh
   # Align multichannel images
   cd image_preprocessing/alignment_testing
   python panchromatic_alignment.py
   
   # Split into train/test sets
   cd ../
   python train_test_split.py
   
   # Apply augmentation if needed
   python img_augmentation.py
   ```

2. **Training:**
   ```sh
   # Single model training
   python train.py --dataroot /path/to/data --name experiment_name [options]
   
   # Or multi-polarization training
   python polarimetric_training.py
   ```

3. **Evaluation:**
   ```sh
   # Flexible evaluation across all trained models
   python flexible_HSP_test-eval.py --base_dir /path/to/results
   
   # Detailed hyperspectral analysis
   python HSI_comparison.py
   ```

### Polarimetric Analysis Workflow
1. **Acquire polarimetric image sequences** (0°, 45°, 90°, 135° polarizer angles)
2. **Run video analysis:**
   ```sh
   cd video_preprocessing
   python thorlabs_video_testing.py
   ```
3. **Train polarimetric models:**
   ```sh
   python polarimetric_training.py
   ```
4. **Evaluate results:**
   ```sh
   python flexible_HSP_test-eval.py --polarization_angles 0 45 90 135
   ```

### Hyperparameter Optimization Workflow
1. **Configure parameter ranges** in `hyperparam_finder.py`
2. **Run automated sweep:**
   ```sh
   cd loss_optimization
   python hyperparam_finder.py
   ```
3. **Analyze results:**
   ```sh
   python analyze_hyperparam_results.py
   ```
4. **Apply best parameters** to production training

## Troubleshooting

### Common Issues
- **Memory errors:** Reduce batch size, use gradient accumulation, or enable mixed precision
- **Alignment failures:** Try different template sizes, check image contrast, verify file formats
- **Training instability:** Adjust learning rates, check loss weights, verify data normalization
- **Poor reconstruction:** Increase model capacity, extend training, check data quality

### Performance Optimization
- Use 16-bit mixed precision training for memory efficiency
- Enable distributed training for multi-GPU setups
- Optimize data loading with appropriate `num_workers`
- Consider model compression for deployment scenarios

## Output Structure
```
results/
├── experiment_name/
│   ├── test_latest/                    # Test results
│   │   ├── images/                     # Generated images
│   │   ├── index.html                  # Web visualization
│   │   └── metrics.csv                 # Quantitative results
│   └── web/                           # Training visualizations
├── evaluation_results/                 # Flexible evaluation outputs
│   ├── detailed_metrics.csv          # Per-image metrics
│   ├── summary_stats.csv             # Aggregate statistics
│   └── visualizations/               # Plots and figures
└── videos/                           # Polarimetric videos
    ├── DoLP_enhanced.mp4             # Degree of Linear Polarization
    └── AoLP_enhanced.mp4             # Angle of Linear Polarization
```

## Tips and References
- For foundational concepts, see the original [CycleGAN and pix2pix documentation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- For hyperspectral-specific evaluation, see the extensive documentation in `HSI_comparison.py`
- For polarimetric analysis details, refer to Stokes parameter calculations in `thorlabs_video_testing.py`
- For alignment techniques, study the comparative approaches in `image_preprocessing/alignment_testing/`
- Use the flexible evaluation script for consistent metric calculation across experiments
- Consider the hyperparameter optimization tools for systematic performance improvement
- Leverage attention mechanisms for improved spectral reconstruction quality

## Contributors and Acknowledgements
- **Core Framework:** Based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- **Hyperspectral Extensions:** Enhanced evaluation metrics and visualization tools
- **Polarimetric Analysis:** Complete DoLP/AoLP calculation and video generation pipeline  
- **Image Alignment:** Multi-modal panchromatic alignment and template matching algorithms
- **Hyperparameter Optimization:** Automated parameter search and analysis framework
- **16-Bit Support:** High bit-depth imaging pipeline and specialized dataloaders

## Citation
If you use this codebase in your research, please consider citing:
```bibtex
@misc{hsp_framework,
  title={Hyperspectral and Polarimetric Image-to-Image Translation Framework},
  author={[Your Name/Institution]},
  year={2024},
  note={Extended from pytorch-CycleGAN-and-pix2pix}
}
```

## License
This project is licensed under the same terms as the original pytorch-CycleGAN-and-pix2pix project. See `LICENSE` file for details.

---

**For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.**
