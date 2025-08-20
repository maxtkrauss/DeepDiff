import os
import random
import shutil
import time

def split_paired_dataset(cubert_dir, thorlabs_dir, output_base, train_ratio=0.8, seed=42, progress_interval=50):
    start_time = time.time()
    random.seed(seed)

    # Get sorted list of files that exist in both directories
    cubert_files = sorted([f for f in os.listdir(cubert_dir) if f.lower().endswith('.tif')])
    thorlabs_files = sorted([f for f in os.listdir(thorlabs_dir) if f.lower().endswith('.tif')])

    # Only keep files present in BOTH
    common_files = sorted(list(set(cubert_files) & set(thorlabs_files)))

    print(f"[INFO] Found {len(common_files)} paired images.")
    if not common_files:
        print("[ERROR] No matching filenames found. Check directory paths.")
        return

    # Shuffle for random split
    random.shuffle(common_files)

    # Split into train/val
    split_idx = int(len(common_files) * train_ratio)
    train_files = common_files[:split_idx]
    val_files = common_files[split_idx:]

    print(f"[INFO] Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_base, split, "cubert"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "thorlabs"), exist_ok=True)

    def move_pairs(file_list, split):
        split_start = time.time()
        for i, fname in enumerate(file_list, start=1):
            src_cubert = os.path.join(cubert_dir, fname)
            src_thorlabs = os.path.join(thorlabs_dir, fname)
            dst_cubert = os.path.join(output_base, split, "cubert", fname)
            dst_thorlabs = os.path.join(output_base, split, "thorlabs", fname)

            shutil.move(src_cubert, dst_cubert)
            shutil.move(src_thorlabs, dst_thorlabs)

            if i <= 3:  # show first few for debugging
                print(f"[DEBUG] Example {split} pair moved: {fname}")
            if i % progress_interval == 0:
                elapsed = time.time() - split_start
                print(f"[INFO] Moved {i}/{len(file_list)} {split} pairs in {elapsed:.1f} sec")

    print("[INFO] Moving train set...")
    move_pairs(train_files, "train")
    print("[INFO] Moving val set...")
    move_pairs(val_files, "val")

    total_elapsed = time.time() - start_time
    print(f"[DONE] Dataset split complete in {total_elapsed:.1f} sec")

if __name__ == "__main__":
    cubert_dir = r"/scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes_raw/cubert"
    thorlabs_dir = r"/scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes_raw/thorlabs"
    output_base = r"/scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes_raw/split"

    split_paired_dataset(cubert_dir, thorlabs_dir, output_base, train_ratio=0.8, seed=42)
