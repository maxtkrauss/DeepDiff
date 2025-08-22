import os
import tifffile
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Folder setup ---
folder_cubert = r"D:\fakenotes_8-14\cropped\cubert"
folder_thorlabs = r"D:\fakenotes_8-14\cropped\thorlabs"

# --- Helper functions ---
def get_all_matching_pairs(cubert_list, thorlabs_list):
    pairs = []
    for cubert_name in cubert_list:
        if cubert_name.endswith("_cubert.tif"):
            base = cubert_name.replace("_cubert.tif", "")
            thorlabs_name = f"{base}_thorlabs.tif"
            if thorlabs_name in thorlabs_list:
                pairs.append((cubert_name, thorlabs_name))
    return pairs


# Normalize to [0, 1] for 12-bit images
def normalize_12bit(img):
    return img.astype(np.float32) / 4095.0

# --- Main processing ---
contents_cubert = os.listdir(folder_cubert)
contents_thorlabs = os.listdir(folder_thorlabs)
matching_pairs = get_all_matching_pairs(contents_cubert, contents_thorlabs)

if not matching_pairs:
    print("No matching TIFF pairs found.")
    exit()

# Only process the first matching pair
cubert_file, thorlabs_file = matching_pairs[0]
cubert_path = os.path.join(folder_cubert, cubert_file)
thorlabs_path = os.path.join(folder_thorlabs, thorlabs_file)
cubert_img = tifffile.imread(cubert_path)
thorlabs_img = tifffile.imread(thorlabs_path)
cubert_panchro = normalize_12bit(np.mean(cubert_img, axis=0))
thorlabs_ref = normalize_12bit(thorlabs_img[0] if thorlabs_img.ndim == 3 else thorlabs_img)
print(f"Pair: {cubert_file} ({cubert_img.shape}), {thorlabs_file} ({thorlabs_img.shape})")

# 2. Downsample Thorlabs Ch 0 to Cubert spatial size for feature matching
thorlabs_ref = thorlabs_img[0] if thorlabs_img.ndim == 3 else thorlabs_img
thorlabs_resized = cv2.resize(thorlabs_ref, cubert_panchro.shape[::-1], interpolation=cv2.INTER_AREA)

# 3. SIFT feature detection and matching (at low-res)
sift = cv2.SIFT_create()
cubert_panchro_uint8 = cv2.normalize(cubert_panchro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
thorlabs_resized_uint8 = cv2.normalize(thorlabs_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

kp1, des1 = sift.detectAndCompute(cubert_panchro_uint8, None)
kp2, des2 = sift.detectAndCompute(thorlabs_resized_uint8, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des2, des1, k=2)  # thorlabs to cubert

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) >= 4:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # Find affine transform (or homography for more flexibility)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    # --- Scale affine transform to native Thorlabs resolution ---
    scale_x = thorlabs_ref.shape[1] / cubert_panchro.shape[1]
    scale_y = thorlabs_ref.shape[0] / cubert_panchro.shape[0]
    M_native = M.copy()
    M_native[0, 0] *= scale_x
    M_native[0, 1] *= scale_x
    M_native[1, 0] *= scale_y
    M_native[1, 1] *= scale_y
    M_native[0, 2] *= scale_x
    M_native[1, 2] *= scale_y
    # Warp the high-res Thorlabs image to its own native resolution
    thorlabs_aligned_native = cv2.warpAffine(
        thorlabs_ref, M_native, (thorlabs_ref.shape[1], thorlabs_ref.shape[0]), flags=cv2.INTER_LINEAR
    )
    # For difference, upsample Cubert panchro to Thorlabs size
    cubert_panchro_up = cv2.resize(cubert_panchro, thorlabs_ref.shape[::-1], interpolation=cv2.INTER_CUBIC)
    diff_heatmap = np.abs(thorlabs_aligned_native - cubert_panchro_up)
else:
    print("Not enough good matches for SIFT alignment.")
    thorlabs_aligned_native = thorlabs_ref
    cubert_panchro_up = cv2.resize(cubert_panchro, thorlabs_ref.shape[::-1], interpolation=cv2.INTER_CUBIC)
    diff_heatmap = np.abs(thorlabs_aligned_native - cubert_panchro_up)

# For difference, upsample Cubert panchro to Thorlabs size
cubert_panchro_up = cv2.resize(cubert_panchro, thorlabs_ref.shape[::-1], interpolation=cv2.INTER_CUBIC)


# Use the sharpened image for the difference heatmap
diff_heatmap = np.abs(thorlabs_aligned_native - thorlabs_img[0])

# --- Visualization ---
plt.figure(figsize=(25, 4))
plt.subplot(1, 4, 1)
plt.title("Original Cubert Panchro (mean, native)")
plt.imshow(cubert_panchro, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Cubert Panchro (upsampled)")
plt.imshow(cubert_panchro_up, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Aligned Thorlabs (native)")
plt.imshow(thorlabs_aligned_native, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Difference Heatmap")
plt.imshow(diff_heatmap, cmap='hot')
plt.axis('off')

plt.tight_layout()
plt.show()