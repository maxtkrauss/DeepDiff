import tifffile
import numpy as np

import matplotlib.pyplot as plt

# Path to the 5-channel TIFF file
tif_path = r"D:\banknotes_4-14\pol_sorted\0_degree\validation\thorlabs"

# List of TIFF file paths
tif_paths = [
    r"D:\banknotes_4-14\pol_sorted\0_degree\validation\thorlabs\NZ_processed_0_degree_thorlabs_thorlabs_mirrored.tif",
    r"D:\banknotes_4-14\pol_sorted\0_degree\validation\thorlabs\india_processed_0_degree_thorlabs_thorlabs_mirrored.tif",
    r"D:\banknotes_4-14\pol_sorted\0_degree\validation\thorlabs\NZ_cellophane_processed_0_degree_thorlabs_thorlabs_mirrored.tif"
]

def compute_DoLP_AoLP(img):
    # Assumes img shape: (5, H, W), channels: I0, I45, I90, I135, Iunpol
    I0, I45, I90, I135, _ = img
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    DoLP = np.clip(np.sqrt(S1**2 + S2**2) / (S0 + 1e-8), 0, 1)
    AoLP = 0.5 * np.arctan2(S2, S1)
    return DoLP, AoLP

for tif_path in tif_paths:
    img = tifffile.imread(tif_path)
    DoLP, AoLP = compute_DoLP_AoLP(img)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(DoLP, cmap='gray')
    axes[0].set_title('DoLP')
    axes[0].axis('off')
    im = axes[1].imshow(AoLP, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
    axes[1].set_title('AoLP')
    axes[1].axis('off')
    plt.suptitle(tif_path.split("\\")[-1])
    plt.tight_layout()
    plt.show()

# Load the TIFF file (channels in the first dimension)
img = tifffile.imread(tif_path)

# Plot the first four channels
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(img[i], cmap='gray')
    axes[i].set_title(f'Channel {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()