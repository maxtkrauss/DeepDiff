from PIL import Image
import numpy as np
import tifffile as tiff
import os
import cv2
import matplotlib.pyplot as plt

# List of image paths
image_paths = [
    r"D:\objects_4-24\processed_fruit_videos\polarized1\validation\thorlabs\frame_0000.tif", # Fruit 1
    r"D:\objects_4-24\processed_fruit_videos\polarized2\validation\thorlabs\frame_0403.tif", # Fruit 2
    r"D:\objects_4-24\processed_fruit_videos\polarized3\validation\thorlabs\frame_0649.tif", # Fruit 3
]

# Prepare figure: 1 row per image, 2 columns (DoLP, AoLP)
fig, axes = plt.subplots(len(image_paths), 2, figsize=(10, 5 * len(image_paths)))

if len(image_paths) == 1:
    axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D

for idx, image_path in enumerate(image_paths):
    img_array = tiff.imread(image_path)
    print(f"Image {idx+1} shape:", img_array.shape)

    # Extract polarization channels (assuming channels 0-3 are 0°, 45°, 90°, 135°)
    I0 = img_array[0].astype(np.float32)
    I45 = img_array[1].astype(np.float32)
    I90 = img_array[2].astype(np.float32)
    I135 = img_array[3].astype(np.float32)

    # Calculate Stokes parameters
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    # Degree of Linear Polarization (DoLP)
    DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
    # Normalize DoLP to [0, 1]
    DoLP = np.clip(DoLP, 0, 1)

    # Angle of Linear Polarization (AoLP), in radians
    AoLP = 0.5 * np.arctan2(S2, S1)

    im0 = axes[idx, 0].imshow(DoLP, cmap='gray', vmin=0, vmax=1)
    axes[idx, 0].set_title(f'Fruit {idx+1} DoLP')
    axes[idx, 0].axis('off')
    plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

    im1 = axes[idx, 1].imshow(AoLP, cmap='hsv')
    axes[idx, 1].set_title(f'Fruit {idx+1} AoLP (radians)')
    axes[idx, 1].axis('off')
    plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()



# each image in this folder is a 5-channel tiff frame. I want to take each frame, compute the DoLP and AoLP, and then reconstruct the video of the entire folder in a DoLP and AoLP video.# The first 4 channels are the polarization channels (0, 45, 90, 135 degrees), and the 5th channel is the intensity channel.
# The DoLP is computed as sqrt((I0 - I90)^2 + (I45 - I135)^2) / (I0 + I90)
# The AoLP is computed as 0.5 * arctan2(I45 - I135, I0 - I90)
# Save the DoLP and AoLP videos as separate files.
# Folder containing the TIFF frames
folder = r"D:\objects_4-24\processed_fruit_videos\polarized1\validation\thorlabs"

# Get sorted list of TIFF files
tiff_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')])

# Initialize lists to hold DoLP and AoLP frames
dolp_frames = []
aolp_frames = []

for fname in tiff_files:
    img_path = os.path.join(folder, fname)
    img_array = tiff.imread(img_path)

    # Extract polarization channels (assuming channels 0-3 are 0°, 45°, 90°, 135°)
    I0 = img_array[0].astype(np.float32)
    I45 = img_array[1].astype(np.float32)
    I90 = img_array[2].astype(np.float32)
    I135 = img_array[3].astype(np.float32)

    # Compute DoLP and AoLP
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
    DoLP = np.clip(DoLP, 0, 1)  # Normalize to [0, 1]
    AoLP = 0.5 * np.arctan2(S2, S1)  # Radians, range [-pi/2, pi/2]

    # For video, scale DoLP to 8-bit grayscale, AoLP to 8-bit HSV
    dolp_8bit = (DoLP * 255).astype(np.uint8)
    # Map AoLP from [-pi/2, pi/2] to [0, 179] for OpenCV HSV hue channel
    aolp_norm = ((AoLP + (np.pi/2)) / np.pi * 179).astype(np.uint8)

    # For AoLP, create HSV image with full saturation and value
    hsv = np.zeros((*AoLP.shape, 3), dtype=np.uint8)
    hsv[..., 0] = aolp_norm  # Hue
    hsv[..., 1] = 255        # Saturation
    hsv[..., 2] = 255        # Value
    aolp_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    dolp_frames.append(dolp_8bit)
    aolp_frames.append(aolp_bgr)

# Get frame size
height, width = dolp_frames[0].shape

# Define video writers
dolp_out = cv2.VideoWriter(os.path.join(folder, 'DoLP_video.avi'),
                           cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height), False)
aolp_out = cv2.VideoWriter(os.path.join(folder, 'AoLP_video.avi'),
                           cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height), True)

# Write frames to video
for d, a in zip(dolp_frames, aolp_frames):
    dolp_out.write(d)
    aolp_out.write(a)

dolp_out.release()
aolp_out.release()
print("DoLP and AoLP videos saved.")
