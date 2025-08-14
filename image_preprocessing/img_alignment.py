import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# --------- CONFIG ---------
THORLABS_DIR = r"D:\fakenotes_8-14\thorlabs"
CUBERT_DIR   = r"D:\fakenotes_8-14\cubert"
OUT_THORLABS = r"D:\fakenotes_8-14\cropped\thorlabs"
OUT_CUBERT   = r"D:\fakenotes_8-14\cropped\cubert"

THOR_CROP_W = 660
THOR_CROP_H = 660
CUBE_CROP_W = 120
CUBE_CROP_H = 120
# --------------------------

os.makedirs(OUT_THORLABS, exist_ok=True)
os.makedirs(OUT_CUBERT,   exist_ok=True)

# ---------- Helpers ----------
def load_multichannel(path):
    """
    Return array as (C, H, W).
    Handles:
      (H, W)          -> (1, H, W)
      (pages, H, W)   -> (pages, H, W)  (kept)
      (H, W, C)       -> (C, H, W)      (moved)
    We avoid any heuristic that would flip a proper (C,H,W) like (106,410,410).
    """
    arr = tifffile.imread(path)
    if arr.ndim == 2:                    # (H, W)
        return arr[None, ...]
    if arr.ndim == 3:
        # If last dim is a plausible channel count, treat as channel-last
        if arr.shape[-1] in (1, 3, 4, 5, 106) or arr.shape[-1] <= 32:
            return np.moveaxis(arr, -1, 0)  # (H,W,C) -> (C,H,W)
        # Otherwise assume already (C,H,W)
        return arr
    raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}")

def clamp_crop(x, y, crop_w, crop_h, H, W):
    return max(0, min(x, W - crop_w)), max(0, min(y, H - crop_h))

def panchro(img_CHW):
    """
    Create a panchromatic (H,W) preview by averaging channels
    with robust percentile normalization to 0..255 uint8.
    """
    pano = img_CHW.mean(axis=0)
    # robust contrast; guard against flat images
    lo, hi = np.percentile(pano, (1, 99))
    denom = max(hi - lo, 1e-9)
    pano = np.clip((pano - lo) / denom, 0, 1)
    return (pano * 255).astype(np.uint8)

def pick_top_left(img2d, crop_w, crop_h, title):
    """
    Interactive: click top-left for a fixed crop (crop_w x crop_h).
    Keys: c=confirm, r=redo, q=quit
    """
    H, W = img2d.shape
    fig, ax = plt.subplots()
    im = ax.imshow(img2d, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')  # prevent stretching
    ax.set_title(f"{title}\nClick top-left for {crop_w}×{crop_h} | c=confirm, r=redo, q=quit")
    rect_artist = None
    state = {'x': None, 'y': None, 'action': None}

    def draw_rect(x0, y0):
        nonlocal rect_artist
        if rect_artist:
            rect_artist.remove()
        rect_artist = ax.add_patch(
            plt.Rectangle((x0, y0), crop_w, crop_h, fill=False, color='red', linewidth=1.5)
        )
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = clamp_crop(int(event.xdata), int(event.ydata), crop_w, crop_h, H, W)
        state['x'], state['y'] = x, y
        draw_rect(x, y)

    def onkey(event):
        if event.key in ('c', 'r', 'q'):
            state['action'] = event.key
            if event.key in ('q', 'r'):
                plt.close(fig)
            elif event.key == 'c' and state['x'] is not None:
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    return state['action'], state['x'], state['y']

def crop_fixed(arrCHW, x, y, w, h):
    return arrCHW[:, y:y+h, x:x+w]

def save_multipage_tiff(path, arrCHW):
    # Save as pages (C,H,W)
    tifffile.imwrite(path, arrCHW, photometric='minisblack')

def extract_idx(name):
    m = re.search(r'image_(\d+)_', name)
    return int(m.group(1)) if m else None
# ----------------------------

# --- Match by index number ---
thor_files = {extract_idx(f): f for f in os.listdir(THORLABS_DIR) if f.lower().endswith('.tif')}
cube_files = {extract_idx(f): f for f in os.listdir(CUBERT_DIR)   if f.lower().endswith('.tif')}
thor_files = {k: v for k, v in thor_files.items() if k is not None}
cube_files = {k: v for k, v in cube_files.items() if k is not None}

common_indices = sorted(set(thor_files.keys()) & set(cube_files.keys()))
print(f"Found {len(common_indices)} paired TIFFs.")

if not common_indices:
    raise RuntimeError("No matching image pairs found.")

# --- Step 1: Pick first pair for crop coordinates (with proper previews) ---
first_idx = common_indices[0]
thor_name = thor_files[first_idx]
cube_name = cube_files[first_idx]
thor_img = load_multichannel(os.path.join(THORLABS_DIR, thor_name))
cube_img = load_multichannel(os.path.join(CUBERT_DIR, cube_name))

print(f"Selecting crop for first pair:\n  Thorlabs: {thor_name}  shape={thor_img.shape}\n  Cubert:   {cube_name}  shape={cube_img.shape}")

# Thorlabs preview (average 5 pol channels)
thor_preview = panchro(thor_img)
action, tx, ty = pick_top_left(thor_preview, THOR_CROP_W, THOR_CROP_H, title="Thorlabs crop")
if action != 'c':
    raise RuntimeError("Crop cancelled for Thorlabs.")

# Cubert preview (average 106 spectral bands) — this fixes the “squished” look
cube_preview = panchro(cube_img)
action, cx, cy = pick_top_left(cube_preview, CUBE_CROP_W, CUBE_CROP_H, title="Cubert crop")
if action != 'c':
    raise RuntimeError("Crop cancelled for Cubert.")

print(f"Thorlabs crop: (x={tx}, y={ty}) size={THOR_CROP_W}x{THOR_CROP_H}")
print(f"Cubert crop:   (x={cx}, y={cy}) size={CUBE_CROP_W}x{CUBE_CROP_H}")

# --- Step 2: Apply to all pairs ---
for i, idx in enumerate(common_indices, 1):
    thor_name = thor_files[idx]
    cube_name = cube_files[idx]
    thor_path = os.path.join(THORLABS_DIR, thor_name)
    cube_path = os.path.join(CUBERT_DIR, cube_name)
    out_thor  = os.path.join(OUT_THORLABS, thor_name)
    out_cube  = os.path.join(OUT_CUBERT,   cube_name)

    thor = load_multichannel(thor_path)
    cube = load_multichannel(cube_path)

    # Sanity: ensure crop fits each image
    _, Ht, Wt = thor.shape
    _, Hc, Wc = cube.shape
    if not (0 <= tx <= Wt - THOR_CROP_W and 0 <= ty <= Ht - THOR_CROP_H):
        print(f"[{i}] WARN: Thorlabs crop out of bounds for {thor_name}; skipping.")
        continue
    if not (0 <= cx <= Wc - CUBE_CROP_W and 0 <= cy <= Hc - CUBE_CROP_H):
        print(f"[{i}] WARN: Cubert crop out of bounds for {cube_name}; skipping.")
        continue

    thor_crop = crop_fixed(thor, tx, ty, THOR_CROP_W, THOR_CROP_H)
    cube_crop = crop_fixed(cube, cx, cy, CUBE_CROP_W, CUBE_CROP_H)

    save_multipage_tiff(out_thor, thor_crop)
    save_multipage_tiff(out_cube, cube_crop)
    print(f"[{i}/{len(common_indices)}] Saved crops: {thor_name} | {cube_name}")
