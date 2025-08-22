# Files paired between banknotes and fakenotes:
# image_71 banknote, image_1 fakenote
# image_68 banknote, image_4 fakenote
# image_18 banknote, image_2 fakenote

# Assorted fakenotes for test set:
# image_25, image_20, image_19

import os
import cv2
import numpy as np
import tifffile as tiff

# Directories for banknotes and fakenotes
fakenotes_dir_cubert = r"D:\fakenotes_8-14\cropped\cubert"
fakenotes_dir_thorlabs = r"D:\fakenotes_8-14\cropped\thorlabs"

banknotes_dir_cubert = r"D:\banknotes_4-15\processed\cumulative\cubert"
banknotes_dir_thorlabs = r"D:\banknotes_4-15\processed\cumulative\thorlabs"

# List all files in each directory
banknote_files_cubert = sorted([f for f in os.listdir(banknotes_dir_cubert) if f.endswith('.tif') or f.endswith('.tiff')])
banknote_files_thorlabs = sorted([f for f in os.listdir(banknotes_dir_thorlabs) if f.endswith('.tif') or f.endswith('.tiff')])
fakenote_files_cubert = sorted([f for f in os.listdir(fakenotes_dir_cubert) if f.endswith('.tif') or f.endswith('.tiff')])
fakenote_files_thorlabs = sorted([f for f in os.listdir(fakenotes_dir_thorlabs) if f.endswith('.tif') or f.endswith('.tiff')])

def extract_image_id(filename):
    # Example: image_12_cubert_cubert.tiff -> 12
    parts = filename.split('_')
    if len(parts) >= 2 and parts[0] == 'image':
        return parts[1]
    return None

# Build dicts for fast lookup
banknote_dict_cubert = {extract_image_id(f): f for f in banknote_files_cubert}
banknote_dict_thorlabs = {extract_image_id(f): f for f in banknote_files_thorlabs}
fakenote_dict_cubert = {extract_image_id(f): f for f in fakenote_files_cubert}
fakenote_dict_thorlabs = {extract_image_id(f): f for f in fakenote_files_thorlabs}

# Find pairs by matching image IDs within cubert and within thorlabs
paired_banknotes = []
for img_id in banknote_dict_cubert:
    if img_id in banknote_dict_thorlabs:
        paired_banknotes.append((
            os.path.join(banknotes_dir_cubert, banknote_dict_cubert[img_id]),
            os.path.join(banknotes_dir_thorlabs, banknote_dict_thorlabs[img_id])
        ))

paired_fakenotes = []
for img_id in fakenote_dict_cubert:
    if img_id in fakenote_dict_thorlabs:
        paired_fakenotes.append((
            os.path.join(fakenotes_dir_cubert, fakenote_dict_cubert[img_id]),
            os.path.join(fakenotes_dir_thorlabs, fakenote_dict_thorlabs[img_id])
        ))

print(f"Number of paired banknote images (cubert/thorlabs): {len(paired_banknotes)}")
print(f"Number of paired fakenote images (cubert/thorlabs): {len(paired_fakenotes)}")

# Create output directory
output_dir = r"D:\currency_dataset"
cubert_out_dir = os.path.join(output_dir, "cubert")
thorlabs_out_dir = os.path.join(output_dir, "thorlabs")
os.makedirs(cubert_out_dir, exist_ok=True)
os.makedirs(thorlabs_out_dir, exist_ok=True)

def copy_and_label(pairs, label):
    for idx, (cubert_path, thorlabs_path) in enumerate(pairs):
        img_id = extract_image_id(os.path.basename(cubert_path))
        # Copy cubert image
        cubert_dst = os.path.join(cubert_out_dir, f"{label}_{img_id}_cubert.tiff")
        tiff.imwrite(cubert_dst, tiff.imread(cubert_path))
        # Copy thorlabs image
        thorlabs_dst = os.path.join(thorlabs_out_dir, f"{label}_{img_id}_thorlabs.tiff")
        tiff.imwrite(thorlabs_dst, tiff.imread(thorlabs_path))

copy_and_label(paired_banknotes, "banknote")
copy_and_label(paired_fakenotes, "fakenote")

print(f"Combined dataset written to {output_dir}")
