import os
import re
import random
from PIL import Image, ImageOps

def augment_image_frames(frames, seed):
    random.seed(seed)
    aug_frames = []

    # Choose transforms once per seed so all frames get the same
    do_hflip = random.choice([True, False])
    do_vflip = random.choice([True, False])
    angle = random.choice([0, 90, 180, 270])

    for frame in frames:
        img = frame.copy()
        if do_hflip:
            img = ImageOps.mirror(img)
        if do_vflip:
            img = ImageOps.flip(img)
        if angle:
            img = img.rotate(angle, expand=True)
        aug_frames.append(img)
    return aug_frames

def parse_id_and_suffix(filename):
    match = re.match(r'image_(\d+)_.*_(.*)\.tif$', filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None

def get_image_pairs(dir1, dir2):
    files1 = [f for f in os.listdir(dir1) if f.lower().endswith('.tif')]
    files2 = [f for f in os.listdir(dir2) if f.lower().endswith('.tif')]

    dict1 = {parse_id_and_suffix(f): f for f in files1 if parse_id_and_suffix(f)}
    dict2 = {parse_id_and_suffix(f): f for f in files2 if parse_id_and_suffix(f)}

    common_keys = set(dict1.keys()) & set(dict2.keys())
    return [
        (os.path.join(dir1, dict1[k]), os.path.join(dir2, dict2[k]))
        for k in common_keys
    ]

def augment_folder(base_dir, num_aug=20):
    cubert_dir = os.path.join(base_dir, "cubert")
    thorlabs_dir = os.path.join(base_dir, "thorlabs")
    aug_cubert_dir = os.path.join(base_dir, "augmented_cubert")
    aug_thorlabs_dir = os.path.join(base_dir, "augmented_thorlabs")
    os.makedirs(aug_cubert_dir, exist_ok=True)
    os.makedirs(aug_thorlabs_dir, exist_ok=True)

    pairs = get_image_pairs(cubert_dir, thorlabs_dir)
    print(f"Found {len(pairs)} pairs in {base_dir}")

    counter = 0
    for idx, (cubert_path, thorlabs_path) in enumerate(pairs):
        # Open as list of frames (multi-page TIFF)
        cubert_img = Image.open(cubert_path)
        cubert_frames = []
        try:
            while True:
                cubert_frames.append(cubert_img.copy())
                cubert_img.seek(cubert_img.tell() + 1)
        except EOFError:
            pass

        thorlabs_img = Image.open(thorlabs_path)
        thorlabs_frames = []
        try:
            while True:
                thorlabs_frames.append(thorlabs_img.copy())
                thorlabs_img.seek(thorlabs_img.tell() + 1)
        except EOFError:
            pass

        base_name = os.path.splitext(os.path.basename(cubert_path))[0]

        for i in range(num_aug):
            seed = hash((base_name, i))
            cubert_aug_frames = augment_image_frames(cubert_frames, seed)
            thorlabs_aug_frames = augment_image_frames(thorlabs_frames, seed)

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

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            print(f"Processed {idx + 1}/{len(pairs)} pairs in {base_dir} (total {counter} augmentations)")

if __name__ == "__main__":
    train_dir = r"/scratch/general/nfs1/u1528328/img_dir/wwalker/bigtester/Upload_All/train/"
    test_dir  = r"/scratch/general/nfs1/u1528328/img_dir/wwalker/bigtester/Upload_All/test/"

    augment_folder(train_dir)
    augment_folder(test_dir)
