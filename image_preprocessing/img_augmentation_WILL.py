import os
import re
from PIL import Image, ImageOps

def augment_image_frames_all_combinations(frames):
    # All combinations of hflip, vflip, and rotation (4 angles)
    combinations = []
    for do_hflip in [False, True]:
        for do_vflip in [False, True]:
            for angle in [0, 90, 180, 270]:
                combinations.append((do_hflip, do_vflip, angle))
    aug_frames_list = []
    for do_hflip, do_vflip, angle in combinations:
        aug_frames = []
        for frame in frames:
            img = frame.copy()
            if do_hflip:
                img = ImageOps.mirror(img)
            if do_vflip:
                img = ImageOps.flip(img)
            if angle:
                img = img.rotate(angle, expand=True)
            aug_frames.append(img)
        aug_frames_list.append(aug_frames)
    return aug_frames_list

def parse_id_and_suffix(filename):
    import re
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

def augment_folder(base_dir):
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

        cubert_base = os.path.splitext(os.path.basename(cubert_path))[0]
        thorlabs_base = os.path.splitext(os.path.basename(thorlabs_path))[0]


        cubert_aug_frames_list = augment_image_frames_all_combinations(cubert_frames)
        thorlabs_aug_frames_list = augment_image_frames_all_combinations(thorlabs_frames)

        for i, (cubert_aug_frames, thorlabs_aug_frames) in enumerate(zip(cubert_aug_frames_list, thorlabs_aug_frames_list)):
            cubert_aug_frames[0].save(
                os.path.join(aug_cubert_dir, f"{cubert_base}_aug{i}.tif"),
                save_all=True,
                append_images=cubert_aug_frames[1:]
            )
            thorlabs_aug_frames[0].save(
                os.path.join(aug_thorlabs_dir, f"{thorlabs_base}_aug{i}.tif"),
                save_all=True,
                append_images=thorlabs_aug_frames[1:]
            )
            counter += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            print(f"Processed {idx + 1}/{len(pairs)} pairs in {base_dir} (total {counter} augmentations)")

if __name__ == "__main__":
    train_dir = r"/scratch/general/nfs1/u1528328/img_dir/wwalker/bigtester/Upload_All/"
    augment_folder(os.path.dirname(train_dir))
