import os
import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt

# --- Single Image Visualization ---

image_paths = [
    r"D:\objects_4-24\processed_fruit_videos\polarized1\validation\thorlabs\frame_0000.tif",
    r"D:\objects_4-24\processed_fruit_videos\polarized2\validation\thorlabs\frame_0403.tif",
    r"D:\objects_4-24\processed_fruit_videos\polarized3\validation\thorlabs\frame_0649.tif",
]

# List of image paths
for idx, image_path in enumerate(image_paths):
    img_array = tiff.imread(image_path)
    print(f"Image {idx+1} shape:", img_array.shape)

    I0 = img_array[0].astype(np.float32)
    I45 = img_array[1].astype(np.float32)
    I90 = img_array[2].astype(np.float32)
    I135 = img_array[3].astype(np.float32)

    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
    DoLP = np.clip(DoLP, 0, 1)
    AoLP = 0.5 * np.arctan2(S2, S1)  # [-pi/2, pi/2]

    # Show DoLP
    plt.figure(figsize=(5, 5))
    im0 = plt.imshow(DoLP, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Fruit {idx+1} DoLP')
    plt.axis('off')
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Show AoLP
    plt.figure(figsize=(5, 5))
    im1 = plt.imshow(AoLP, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
    plt.title(f'Fruit {idx+1} AoLP (radians)')
    plt.axis('off')
    cbar = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi/2, 0, np.pi/2])
    cbar.set_ticklabels([r'$-\pi/2$', '0', r'$\pi/2$'])
    plt.tight_layout()
    plt.show()
