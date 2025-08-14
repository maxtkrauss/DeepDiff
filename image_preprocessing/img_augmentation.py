import os
from PIL import Image, ImageSequence
import numpy as np
from tqdm import tqdm
import pandas as pd

# Base directories
# base_dir = r"D:\banknotes_4-15\processed\training" # for local testing
base_dir = r"/scratch/general/nfs1/u1528328/img_dir/wwalker/bigtester/Upload_All/test"  
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
    return [
        (0, False, False),
        (90, False, False),
        (180, False, False),
        (270, False, False),
        (0, True, False),
        (90, True, False),
        (180, True, False),
        (270, True, False)
    ]

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

def apply_mask_to_pairs(cubert_frames, thorlabs_frames, mask_fraction=0.25):
    """
    Zero out a matching square region in both cubert and thorlabs image stacks.
    """
    # Convert PIL to numpy (C, H, W)
    cubert_np = np.stack([np.array(f) for f in cubert_frames])  # (C, H, W)
    thorlabs_np = np.stack([np.array(f) for f in thorlabs_frames])  # (1, H, W)

    C_c, H_c, W_c = cubert_np.shape
    C_t, H_t, W_t = thorlabs_np.shape

    # Use normalized coordinates for matching locations
    top_frac = np.random.uniform(0.0, 1.0 - np.sqrt(mask_fraction))
    left_frac = np.random.uniform(0.0, 1.0 - np.sqrt(mask_fraction))

    mask_size_c = int(np.sqrt(mask_fraction) * H_c)
    mask_size_t = int(np.sqrt(mask_fraction) * H_t)

    top_c = int(top_frac * H_c)
    left_c = int(left_frac * W_c)
    top_t = int(top_frac * H_t)
    left_t = int(left_frac * W_t)

    cubert_np[:, top_c:top_c+mask_size_c, left_c:left_c+mask_size_c] = 0
    thorlabs_np[:, top_t:top_t+mask_size_t, left_t:left_t+mask_size_t] = 0

    # Convert back to PIL format
    cubert_masked = [Image.fromarray(cubert_np[i].astype(np.uint16)) for i in range(C_c)]
    thorlabs_masked = [Image.fromarray(thorlabs_np[i].astype(np.uint16)) for i in range(C_t)]

    return cubert_masked, thorlabs_masked

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

    # for mask in [False, True]:
    #     for i, (angle, flip_h, flip_v) in enumerate(transforms_list):
    #         aug_index = i + (8 if mask else 0)  # offset index for masked versions

    #         cubert_aug_frames = apply_transform_to_all_frames(cubert_img, angle, flip_h, flip_v)
    #         thorlabs_aug_frames = apply_transform_to_all_frames(thorlabs_img, angle, flip_h, flip_v)

    #         # Apply synchronized masking if mask == True
    #         if mask:
    #             cubert_aug_frames, thorlabs_aug_frames = apply_mask_to_pairs(
    #                 cubert_aug_frames, thorlabs_aug_frames, mask_fraction=mask_fraction
    #             )

    #         # Save the augmented image stack
    #         cubert_aug_frames[0].save(
    #             os.path.join(aug_cubert_dir, f"{base_name}_aug{aug_index}.tif"),
    #             save_all=True,
    #             append_images=cubert_aug_frames[1:]
    #         )
    #         thorlabs_aug_frames[0].save(
    #             os.path.join(aug_thorlabs_dir, f"{base_name}_aug{aug_index}.tif"),
    #             save_all=True,
    #             append_images=thorlabs_aug_frames[1:]
    #         )
    #         counter += 1

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
