import numpy as np
import tifffile
import matplotlib.pyplot as plt

real_note_1 = r"D:\banknotes_4-15\banknotes\validation\cubert\image_38_aug0.tif"
real_note_2 = r"D:\banknotes_4-15\banknotes\validation\cubert\image_68_aug0.tif"
real_note_3 = r"D:\banknotes_4-15\banknotes\validation\cubert\image_67_aug0.tif"
fake_note_1_path = r"D:\fakenotes_8-14\cropped\cubert\image_1_cubert.tif"  # rotate 180 (vertical flip)
fake_note_2_path = r"D:\fakenotes_8-14\cropped\cubert\image_4_cubert.tif"  # rotate 180 (vertical flip)
fake_note_3_path = r"D:\fakenotes_8-14\cropped\cubert\image_6_cubert.tif"  # mirror across vertical axis (horizontal flip)

def load_selected_channels(path, indices, flip_vertical=False, flip_horizontal=False):
    img = tifffile.imread(path)
    img = img[indices]
    if flip_vertical:
        img = np.flip(img, axis=1)
    if flip_horizontal:
        img = np.flip(img, axis=2)
    return img

# Choose 5 wavelengths evenly spaced between 450 and 850 nm
num_channels = 106
wavelengths = np.linspace(450, 850, num_channels)
selected_indices = np.linspace(0, num_channels - 1, 5, dtype=int)
selected_wavelengths = wavelengths[selected_indices]

# Load images
real_imgs = [
    load_selected_channels(real_note_1, selected_indices),
    load_selected_channels(real_note_2, selected_indices),
    load_selected_channels(real_note_3, selected_indices),
]
fake_imgs = [
    load_selected_channels(fake_note_1_path, selected_indices, flip_vertical=True),      # rotate 180
    load_selected_channels(fake_note_2_path, selected_indices, flip_vertical=True),      # rotate 180
    load_selected_channels(fake_note_3_path, selected_indices, flip_horizontal=True),    # mirror across vertical axis
]

# Plotting
for i in range(3):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Note Set {i+1}: Real (top) vs Fake (bottom)', fontsize=16)
    for j in range(5):
        # Real note
        ax = axes[0, j]
        ax.imshow(real_imgs[i][j], cmap='gray')
        ax.set_title(f'{int(selected_wavelengths[j])} nm')
        ax.axis('off')
        # Fake note
        ax = axes[1, j]
        ax.imshow(fake_imgs[i][j], cmap='gray')
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
