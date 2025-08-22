import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# --------- CONFIG ---------
THORLABS_IMAGE = r"D:\spwiders\spider1_0.tif"  # Direct path to your image
OUT_THORLABS = r"D:\fakenotes_8-14\cropped\thorlabs"

THOR_CROP_W = 660
THOR_CROP_H = 660
# --------------------------

os.makedirs(OUT_THORLABS, exist_ok=True)

# ---------- Helpers ----------
def load_multichannel(path):
    """
    Return array as (C, H, W).
    Handles:
      (H, W)          -> (1, H, W)
      (pages, H, W)   -> (pages, H, W)  (kept)
      (H, W, C)       -> (C, H, W)      (moved)
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
    fig, ax = plt.subplots(figsize=(10, 8))
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
            plt.Rectangle((x0, y0), crop_w, crop_h, fill=False, color='red', linewidth=2)
        )
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = clamp_crop(int(event.xdata), int(event.ydata), crop_w, crop_h, H, W)
        state['x'], state['y'] = x, y
        draw_rect(x, y)
        print(f"Selected crop region: top-left=({x}, {y}), size={crop_w}×{crop_h}")

    def onkey(event):
        if event.key in ('c', 'r', 'q'):
            state['action'] = event.key
            if event.key == 'q':
                print("Cropping cancelled.")
                plt.close(fig)
            elif event.key == 'r':
                print("Redoing selection...")
                state['x'], state['y'] = None, None
                if rect_artist:
                    rect_artist.remove()
                    rect_artist = None
                fig.canvas.draw_idle()
            elif event.key == 'c' and state['x'] is not None:
                print(f"Crop confirmed at ({state['x']}, {state['y']})")
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

# ----------------------------

def main():
    # Check if the image exists
    if not os.path.exists(THORLABS_IMAGE):
        raise RuntimeError(f"Image not found: {THORLABS_IMAGE}")
    
    print(f"Loading image: {THORLABS_IMAGE}")
    
    try:
        # Load the image
        thor_img = load_multichannel(THORLABS_IMAGE)
        print(f"Image shape: {thor_img.shape}")
        
        # Create preview (average all channels/frames)
        thor_preview = panchro(thor_img)
        
        # Interactive crop selection
        action, tx, ty = pick_top_left(thor_preview, THOR_CROP_W, THOR_CROP_H, 
                                       title=f"Thorlabs crop - {os.path.basename(THORLABS_IMAGE)}")
        
        if action != 'c':
            print("Crop cancelled or failed.")
            return
        
        print(f"Selected crop region: (x={tx}, y={ty}) size={THOR_CROP_W}x{THOR_CROP_H}")
        
        # Apply crop
        thor_crop = crop_fixed(thor_img, tx, ty, THOR_CROP_W, THOR_CROP_H)
        
        # Save cropped image
        output_name = f"cropped_{os.path.basename(THORLABS_IMAGE)}"
        output_path = os.path.join(OUT_THORLABS, output_name)
        save_multipage_tiff(output_path, thor_crop)
        
        print(f"Cropped image saved to: {output_path}")
        print(f"Cropped image shape: {thor_crop.shape}")
        
    except Exception as e:
        print(f"ERROR processing {THORLABS_IMAGE}: {e}")

if __name__ == "__main__":
    main()