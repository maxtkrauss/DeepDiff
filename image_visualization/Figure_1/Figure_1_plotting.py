import tifffile
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

ground_truth = r"image_visualization\Figure_1\gt_img_cropped.tif"
reconstructed = r"image_visualization\Figure_1\recon_img_cropped.tif"

gt_img = tifffile.imread(ground_truth)
recon_img = tifffile.imread(reconstructed)

print("Ground truth shape:", gt_img.shape)
print("Reconstructed shape:", recon_img.shape)

import matplotlib.pyplot as plt

def panchromatic(img):
    return np.mean(img, axis=0)

def spectral_fidelity(gt, rc):
    dot = np.sum(gt * rc)
    norm_gt = np.sqrt(np.sum(gt ** 2))
    norm_rc = np.sqrt(np.sum(rc ** 2))
    if norm_gt == 0 or norm_rc == 0:
        return 0
    return dot / (norm_gt * norm_rc)

def spectral_relative_error(gt, rc):
    eps = 1e-8
    return 100 * np.mean(np.abs(gt - rc) / (np.abs(gt) + eps))

# Generate panchromatic images
gt_pan = panchromatic(gt_img)
rc_pan = panchromatic(recon_img)

# Compute per-pixel spectral similarity (cosine similarity)
h, w = gt_pan.shape
similarity_map = np.zeros((h, w))
for y in range(h):
    for x in range(w):
        gt_spec = gt_img[:, y, x]
        rc_spec = recon_img[:, y, x]
        similarity_map[y, x] = spectral_fidelity(gt_spec, rc_spec)

# Find coordinates of the top 3 most similar pixels
flat_indices = np.argpartition(similarity_map.flatten(), -3)[-3:]
coords = np.column_stack(np.unravel_index(flat_indices, similarity_map.shape))

wavelengths = np.linspace(450, 850, gt_img.shape[0])

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Show panchromatic images
axs[0].imshow(gt_pan, cmap='gray')
axs[0].set_title('Ground Truth Panchromatic')
axs[1].imshow(rc_pan, cmap='gray')
axs[1].set_title('Reconstructed Panchromatic')

# Mark selected pixels
for i, (y, x) in enumerate(coords):
    axs[0].plot(x, y, 'o', label=f'ROI {i+1}')
    axs[1].plot(x, y, 'o', label=f'ROI {i+1}')
axs[0].legend()
axs[1].legend()

# Plot spectra for selected pixels
for i, (y, x) in enumerate(coords):
    gt_spec = gt_img[:, y, x]
    rc_spec = recon_img[:, y, x]
    ssim_val = ssim(gt_spec, rc_spec, data_range=gt_spec.max() - gt_spec.min())
    fid_val = spectral_fidelity(gt_spec, rc_spec)
    sre_val = spectral_relative_error(gt_spec, rc_spec)
    axs[2].plot(wavelengths, gt_spec, label=f'GT ROI {i+1}')
    axs[2].plot(wavelengths, rc_spec, '--', label=f'RC ROI {i+1}')
    axs[2].text(0.02, 0.95 - i*0.12,
        f'ROI {i+1}: SSIM={ssim_val:.3f}, Fid={fid_val:.3f}, SRE%={sre_val:.2f}',
        transform=axs[2].transAxes, color=f'C{i}', fontsize=10, va='top')

axs[2].set_title('Spectral Comparison (3 ROIs)')
axs[2].set_xlabel('Wavelength (nm)')
axs[2].set_ylabel('Intensity')
axs[2].legend()

# Hide unused subplot
axs[3].axis('off')

plt.tight_layout()
plt.show()
