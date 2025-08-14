import os
from PIL import Image, ImageSequence
import numpy as np
from tqdm import tqdm
import pandas as pd

# Base directories
# base_dir = r"D:\banknotes_4-15\processed\training" # for local testing
base_dir = r"/scratch/general/nfs1/u1528328/img_dir/mkrauss/fruit/processed_fruit_4-24/"  
cubert_dir = os.path.join(base_dir, "cubert")
thorlabs_dir = os.path.join(base_dir, "thorlabs")
output_dir = os.path.join(base_dir, "augmented")

# Output folders
aug_cubert_dir = os.path.join(output_dir, "cubert")
aug_thorlabs_dir = os.path.join(output_dir, "thorlabs")
os.makedirs(aug_cubert_dir, exist_ok=True)
os.makedirs(aug_thorlabs_dir, exist_ok=True)

# === Augmentation Settings ===
mask_fraction = 0.25  # 25% square crop

# Define transforms
def generate_transforms():
    transforms = []
    for angle in [0, 90, 180, 270]:
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                transforms.append((angle, flip_h, flip_v))
    return transforms

def apply_transform_to_all_frames(image, angle, flip_h, flip_v):
    transformed_frames = []
    for frame in ImageSequence.Iterator(image):
        img = frame.copy().rotate(angle, resample=Image.NEAREST, fillcolor=0)
        if flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        transformed_frames.append(img)
    return transformed_frames


# Load image pairs
pair_files = [f for f in os.listdir(cubert_dir) if f.endswith(".tif")]
transforms_list = generate_transforms()

counter = 0
for file in tqdm(pair_files, desc="Processing image pairs", total=len(pair_files)):
    base_name = file.replace("_cubert_cubert.tif", "")

    cubert_path = os.path.join(cubert_dir, file)
    thorlabs_filename = f"{base_name}_thorlabs_thorlabs.tif"
    thorlabs_path = os.path.join(thorlabs_dir, thorlabs_filename)

    # Check if the Thorlabs image exists
    if not os.path.exists(thorlabs_path):
        print(f"[SKIP] Missing Thorlabs image: {thorlabs_path}")
        continue

    cubert_img = Image.open(cubert_path)
    thorlabs_img = Image.open(thorlabs_path)

    for i, (angle, flip_h, flip_v) in enumerate(transforms_list):
        cubert_aug_frames = apply_transform_to_all_frames(cubert_img, angle, flip_h, flip_v)
        thorlabs_aug_frames = apply_transform_to_all_frames(thorlabs_img, angle, flip_h, flip_v)

        cubert_aug_frames[0].save(
            os.path.join(aug_cubert_dir, f"{base_name}_aug{i}.tif"),
            save_all=True,
            append_images=cubert_aug_frames[1:]
        )
        thorlabs_aug_frames[0].save(
            os.path.join(aug_thorlabs_dir, f"{base_name}_aug{i}.tif"),
            save_all=True,
            append_images=thorlabs_aug_frames[1:]
        )
        counter += 1
