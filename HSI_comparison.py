import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from skimage.metrics import structural_similarity as ssim
from RGB.HSI2RGB import HSI2RGB  # Assuming HSI2RGB is properly installed
import re

class HyperspectralAnalyzer:
    def __init__(self, image_dir, num_images=756):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.num_images = num_images
        self.valid_indices = range(num_images)
        
        # Interactive visualization variables
        self.current_index = 0
        self.selected_points = []
        self.lines = []
        self.colors = ['red', 'blue', 'green']
        self.color_index = 0
        self.view_mode = "Single Wavelength"
        
        # Metrics storage
        self.ssim_3d_list = []
        self.ssim_2d_list = []
        self.mse_list = []
        self.mae_list = []
        self.rase_list = []
        self.fidelity_list = []  # New list for spectral fidelity
        self.RSE_list = []
        self.sre_l1_list = []  # New SRE lists
        self.sre_l2_list = []
        self.rsre_l1_list = []
        self.rsre_l2_list = []
        
        # Current images metrics
        self.ssim_3d = 0
        self.ssim_2d = 0
        self.mse = 0
        self.mae = 0
        self.rase = 0
        self.fidelity = 0  # New fidelity metric
        self.RSE = 0
        self.sre_l1 = 0  # New SRE metrics
        self.sre_l2 = 0
        self.rsre_l1 = 0
        self.rsre_l2 = 0
        
        # Load initial images
        self.load_images()
    
    def center_crop(self, image, target_shape=(106, 120, 120)):
        """Center crop the image to the target shape."""
        c, h, w = image.shape
        tc, th, tw = target_shape
        
        # Crop only if dimensions are larger
        if c >= tc and h >= th and w >= tw:
            c_start = (c - tc) // 2
            h_start = (h - th) // 2
            w_start = (w - tw) // 2
            return image[c_start:c_start+tc, h_start:h_start+th, w_start:w_start+tw]
        else:
            raise ValueError(f"Image shape {image.shape} is smaller than target shape {target_shape}.")
    
    def load_images(self, index=None):
        """Load ground truth and reconstructed images for the specified index."""
        if index is not None:
            self.current_index = index
        
        cb_pattern = re.compile(rf"^cb_raw_.*_{self.current_index}\.tif$")
        tl_pattern = re.compile(rf"^tl_gen_.*_{self.current_index}\.tif$")
        
        cb_file = next(f for f in self.image_files if cb_pattern.match(f))
        tl_file = next(f for f in self.image_files if tl_pattern.match(f))
        
        cb_path = os.path.join(self.image_dir, cb_file)
        tl_path = os.path.join(self.image_dir, tl_file)
        
        self.ground_truth = tiff.imread(cb_path)
        self.reconstructed = tiff.imread(tl_path)
        
        # Apply center crop if needed
        if self.ground_truth.shape != (106, 660, 660):
            self.ground_truth = self.center_crop(self.ground_truth)
        
        if self.reconstructed.shape != (106, 660, 660):
            self.reconstructed = self.center_crop(self.reconstructed)
        
        # Generate wavelength array
        self.wavelengths = np.linspace(450, 850, self.ground_truth.shape[0])
        
        # Calculate metrics for the current pair
        self.calculate_image_metrics(self.ground_truth, self.reconstructed)
        
        # Convert to RGB for visualization
        self.gt_rgb = self.convert_to_rgb(self.ground_truth)
        self.rc_rgb = self.convert_to_rgb(self.reconstructed)
        
        # Print information about current images
        print(f"Ground Truth ({cb_file}) shape: {self.ground_truth.shape}")
        print(f"Reconstructed ({tl_file}) shape: {self.reconstructed.shape}")
        print(f"3D SSIM: {self.ssim_3d:.4f}, 2D SSIM (450nm): {self.ssim_2d:.4f}")
        print(f"3D MSE: {self.mse:.6f}, 3D MAE: {self.mae:.6f}")
        print(f"RASE: {self.rase:.4f}%, Fidelity: {self.fidelity:.4f}")
        
        return self.ground_truth, self.reconstructed
    
    def calculate_spectral_fidelity(self, ground_truth, reconstructed):
        """
        Calculate spectral fidelity as defined in the paper:
        
        Fidelity = sum(x_i * x_hat_i) / sqrt(sum(x_i^2) * sum(x_hat_i^2))
        
        where x_i is the ground truth and x_hat_i is the reconstructed spectra.
        This is computed for each pixel location and then averaged.
        """
        # Get shapes
        n_bands, height, width = ground_truth.shape
        
        # Initialize fidelity array for each pixel
        pixel_fidelity = np.zeros((height, width))
        
        # Calculate fidelity for each pixel
        for y in range(height):
            for x in range(width):
                gt_spectrum = ground_truth[:, y, x]
                rc_spectrum = reconstructed[:, y, x]
                
                # Handle zeros to avoid division by zero
                if np.sum(gt_spectrum**2) == 0 or np.sum(rc_spectrum**2) == 0:
                    pixel_fidelity[y, x] = 0
                    continue
                
                # Calculate the dot product
                dot_product = np.sum(gt_spectrum * rc_spectrum)
                
                # Calculate the norms
                gt_norm = np.sqrt(np.sum(gt_spectrum**2))
                rc_norm = np.sqrt(np.sum(rc_spectrum**2))
                
                # Calculate fidelity
                pixel_fidelity[y, x] = dot_product / (gt_norm * rc_norm)
        
        # Return average fidelity across all pixels
        return np.mean(pixel_fidelity)
    
    def calculate_image_metrics(self, ground_truth, reconstructed):
        """Calculate metrics for current image pair without normalization."""
        # Calculate SSIM (3D and 2D)
        self.ssim_3d = ssim(ground_truth, reconstructed, data_range=ground_truth.max() - ground_truth.min(), multichannel=False)
        self.ssim_2d = ssim(ground_truth[0], reconstructed[0], data_range=ground_truth[0].max() - ground_truth[0].min())
        
        # Calculate MSE
        self.mse = np.mean((ground_truth - reconstructed) ** 2)
        
        # Calculate MAE
        self.mae = np.mean(np.abs(ground_truth - reconstructed))
        
        # Calculate RASE
        self.rase = self.compute_rase(ground_truth, reconstructed)
        
        # Calculate spectral fidelity
        self.fidelity = self.calculate_spectral_fidelity(ground_truth, reconstructed)

        # Calculate RSE (your existing relative spectral error)
        self.RSE = self.relative_spectral_error_l1(ground_truth, reconstructed)
        
        # Calculate SRE variants
        self.sre_l1 = self.spectral_reconstruction_error_l1(ground_truth, reconstructed)
        self.sre_l2 = self.spectral_reconstruction_error_l2(ground_truth, reconstructed)
        self.rsre_l1 = self.relative_spectral_reconstruction_error(ground_truth, reconstructed, norm='l1')
        self.rsre_l2 = self.relative_spectral_reconstruction_error(ground_truth, reconstructed, norm='l2')
    
    def compute_rase(self, ground_truth, reconstructed):
        """Compute Relative Average Spectral Error (RASE)."""
        N = ground_truth.shape[0]  # Number of spectral bands
        mse_per_band = np.mean((ground_truth - reconstructed) ** 2, axis=(1, 2))
        mean_spectral_value = np.mean(ground_truth)
        
        if mean_spectral_value == 0:
            return np.nan
        
        rase = (100 / N) * np.sqrt(np.sum(mse_per_band) / (mean_spectral_value ** 2))
        return rase
    
    import numpy as np

    def relative_spectral_error_l1(self, gt_img, recon_img):
        if gt_img.shape != recon_img.shape:
            raise ValueError("Input images must have the same shape")
        
        # Reshape to (bands, N), where N = height * width
        bands, height, width = gt_img.shape
        gt_flat = gt_img.reshape(bands, -1)
        recon_flat = recon_img.reshape(bands, -1)
        
        # Compute L1 norm per pixel
        l1_diff = np.abs(gt_flat - recon_flat).sum(axis=0)
        l1_gt = np.abs(gt_flat).sum(axis=0)
        
        # Avoid division by zero
        eps = 1e-8
        relative_errors = l1_diff / (l1_gt + eps)
        
        # Return mean relative error over all pixels
        return relative_errors.mean()

    def spectral_reconstruction_error_l1(self, gt_img, recon_img):
        """
        Calculate Spectral Reconstruction Error using L1 norm.
        SRE_L1 = mean(|GT - Reconstructed|) across all pixels and bands
        """
        if gt_img.shape != recon_img.shape:
            raise ValueError("Input images must have the same shape")
        
        # Calculate absolute difference
        abs_diff = np.abs(gt_img - recon_img)
        
        # Return mean absolute error across all pixels and bands
        return np.mean(abs_diff)

    def spectral_reconstruction_error_l2(self, gt_img, recon_img):
        """
        Calculate Spectral Reconstruction Error using L2 norm (MSE-based).
        SRE_L2 = sqrt(mean((GT - Reconstructed)^2)) across all pixels and bands
        """
        if gt_img.shape != recon_img.shape:
            raise ValueError("Input images must have the same shape")
        
        # Calculate squared difference
        squared_diff = (gt_img - recon_img) ** 2
        
        # Return root mean squared error
        return np.sqrt(np.mean(squared_diff))

    def relative_spectral_reconstruction_error(self, gt_img, recon_img, norm='l1'):
        """
        Calculate Relative Spectral Reconstruction Error (normalized by GT magnitude).
        RSRE = SRE / mean(|GT|) * 100%
        """
        if gt_img.shape != recon_img.shape:
            raise ValueError("Input images must have the same shape")
        
        # Calculate SRE
        if norm == 'l1':
            sre = self.spectral_reconstruction_error_l1(gt_img, recon_img)
            gt_magnitude = np.mean(np.abs(gt_img))
        elif norm == 'l2':
            sre = self.spectral_reconstruction_error_l2(gt_img, recon_img)
            gt_magnitude = np.sqrt(np.mean(gt_img ** 2))
        else:
            raise ValueError("norm must be 'l1' or 'l2'")
        
        if gt_magnitude == 0:
            return np.nan
        
        # Return as percentage
        return (sre / gt_magnitude) * 100

    
    def convert_to_rgb(self, hsi_image):
        """Convert hyperspectral data to RGB using HSI2RGB."""
        # Reshape to (height, width, bands) for compatibility
        hsi_image = hsi_image.transpose(1, 2, 0)
        
        # Define wavelengths
        wl = np.linspace(450, 850, hsi_image.shape[-1])
        
        # Reshape for HSI2RGB processing
        data = np.reshape(hsi_image, (-1, hsi_image.shape[-1]))
        
        # Convert to RGB
        rgb_image = HSI2RGB(wl, data, hsi_image.shape[0], hsi_image.shape[1], 65, 0.002)
        return np.clip(rgb_image, 0, 1)
    
    def calculate_all_metrics(self):
        """Calculate metrics for all test images and display averages."""
        self.ssim_3d_list = []
        self.ssim_2d_list = []
        self.mse_list = []
        self.mae_list = []
        self.rase_list = []
        self.fidelity_list = []  # Add fidelity list
        self.RSE_list = []
        self.sre_l1_list = []  # New SRE lists
        self.sre_l2_list = []
        self.rsre_l1_list = []
        self.rsre_l2_list = []
        
        for index in self.valid_indices:
            try:
                ground_truth, reconstructed = self.load_images(index)
                
                # Store metrics for the current pair
                self.ssim_3d_list.append(self.ssim_3d)
                self.ssim_2d_list.append(self.ssim_2d)
                self.mse_list.append(self.mse)
                self.mae_list.append(self.mae)
                self.rase_list.append(self.rase)
                self.fidelity_list.append(self.fidelity)  # Add fidelity
                self.RSE_list.append(self.RSE)
                self.sre_l1_list.append(self.sre_l1)  # Store SRE metrics
                self.sre_l2_list.append(self.sre_l2)
                self.rsre_l1_list.append(self.rsre_l1)
                self.rsre_l2_list.append(self.rsre_l2)
                
            except FileNotFoundError as e:
                print(e)
                continue
            except StopIteration:
                print(f"No files found for index {index}")
                continue
        
        # Calculate and display average metrics
        avg_ssim_3d = np.mean(self.ssim_3d_list) if self.ssim_3d_list else 0
        avg_ssim_2d = np.mean(self.ssim_2d_list) if self.ssim_2d_list else 0
        avg_mse = np.mean(self.mse_list) if self.mse_list else 0
        avg_mae = np.mean(self.mae_list) if self.mae_list else 0
        avg_rase = np.mean(self.rase_list) if self.rase_list else 0
        avg_fidelity = np.mean(self.fidelity_list) if self.fidelity_list else 0  # Add average fidelity
        avg_RSE = np.mean(self.RSE_list) if self.RSE_list else 0 
        avg_sre_l1 = np.mean(self.sre_l1_list) if self.sre_l1_list else 0  # New SRE averages
        avg_sre_l2 = np.mean(self.sre_l2_list) if self.sre_l2_list else 0
        avg_rsre_l1 = np.mean(self.rsre_l1_list) if self.rsre_l1_list else 0
        avg_rsre_l2 = np.mean(self.rsre_l2_list) if self.rsre_l2_list else 0 
        
        print(f"\nAverage Metrics Over {len(self.ssim_3d_list)} Valid Image Pairs:")
        print(f"  Average 3D SSIM: {avg_ssim_3d:.4f}")
        print(f"  Average 2D SSIM: {avg_ssim_2d:.4f}")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  Average RASE: {avg_rase:.4f}%")
        print(f"  Average Spectral Fidelity: {avg_fidelity:.4f}")  # Print average fidelity
        print(f"  Average RSE: {avg_RSE:.4f}") 
        print(f"  Average SRE (L1): {avg_sre_l1:.6f}")  # New SRE outputs
        print(f"  Average SRE (L2): {avg_sre_l2:.6f}")
        print(f"  Average RSRE (L1): {avg_rsre_l1:.4f}%")
        print(f"  Average RSRE (L2): {avg_rsre_l2:.4f}%") 
        
        return {
            "avg_ssim_3d": avg_ssim_3d,
            "avg_ssim_2d": avg_ssim_2d,
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "avg_rase": avg_rase,
            "avg_fidelity": avg_fidelity,  # Add to return dict
            "avg_RSE" : avg_RSE,
            "avg_sre_l1": avg_sre_l1,  # New SRE returns
            "avg_sre_l2": avg_sre_l2,
            "avg_rsre_l1": avg_rsre_l1,
            "avg_rsre_l2": avg_rsre_l2,
            "num_images": len(self.ssim_3d_list)
        }
    
    def calculate_spectral_fidelity_for_pixel(self, x, y):
        """Calculate spectral fidelity for a specific pixel."""
        gt_spectrum = self.ground_truth[:, y, x]
        rc_spectrum = self.reconstructed[:, y, x]
        
        # Handle zeros to avoid division by zero
        if np.sum(gt_spectrum**2) == 0 or np.sum(rc_spectrum**2) == 0:
            return 0
        
        # Calculate the dot product
        dot_product = np.sum(gt_spectrum * rc_spectrum)
        
        # Calculate the norms
        gt_norm = np.sqrt(np.sum(gt_spectrum**2))
        rc_norm = np.sqrt(np.sum(rc_spectrum**2))
        
        # Calculate fidelity
        return dot_product / (gt_norm * rc_norm)
    
    def calculate_MAE_for_pixel(self, x, y):
        """Calculate spectral fidelity for a specific pixel."""
        gt_spectrum = self.ground_truth[:, y, x]
        rc_spectrum = self.reconstructed[:, y, x]
        return(np.mean(np.abs(gt_spectrum - rc_spectrum)))
    
    def setup_interactive_plot(self):
        """Set up interactive plot for visualization."""
        # Setup figure and axes
        self.fig, self.ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Add dropdown for view mode selection in the bottom-left
        ax_radio = plt.axes([0.02, 0.02, 0.1, 0.15])
        self.radio = RadioButtons(ax_radio, ['RGB Reconstruction', 'Single Wavelength'])
        self.radio.on_clicked(self.change_view_mode)
        
        # Connect click events to the plot
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        
        # Display the initial plot
        self.update_plot()
        
        # Add buttons below the figure
        button_ax_next = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn_next = Button(button_ax_next, "Next")
        self.btn_next.on_clicked(self.load_next_image)
        
        button_ax_clear = plt.axes([0.6, 0.02, 0.2, 0.05])
        self.btn_clear = Button(button_ax_clear, "Clear Spectra")
        self.btn_clear.on_clicked(self.clear_spectra)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()
    
    def change_view_mode(self, label):
        """Handle changes in the view mode."""
        self.view_mode = label
        self.update_plot()
    
    def update_plot(self):
        """Update the plot based on current view mode and data."""
        self.ax[0].cla()
        self.ax[1].cla()
        
        # Update plots based on the selected view mode
        if self.view_mode == "RGB Reconstruction":
            self.ax[0].imshow(self.gt_rgb)
            self.ax[0].set_title("Ground Truth (RGB Reconstruction)")
            self.ax[1].imshow(self.rc_rgb)
            self.ax[1].set_title("Reconstructed (RGB Reconstruction)")
        elif self.view_mode == "Single Wavelength":
            self.ax[0].imshow(self.ground_truth[0], cmap='viridis')
            self.ax[0].set_title("Ground Truth (450nm)")
            self.ax[1].imshow(self.reconstructed[0], cmap='viridis')
            self.ax[1].set_title("Reconstructed (450nm)")
        
        self.ax[0].axis('off')
        self.ax[1].axis('off')
        
        self.ax[2].cla()
        self.ax[2].set_title("Spectral Comparison")
        self.ax[2].set_xlabel("Wavelength (nm)")
        self.ax[2].set_ylabel("Intensity")
        
        # Update figure title with the metrics values including fidelity
        self.fig.suptitle(
            f"Image Analysis: Ground Truth vs Reconstruction (Index {self.current_index}) | "
            f"3D SSIM: {self.ssim_3d:.4f} | 2D SSIM: {self.ssim_2d:.4f} | "
            f"MSE: {self.mse:.6f} | MAE: {self.mae:.6f} | "
            f"RASE: {self.rase:.4f}% | Fidelity: {self.fidelity:.4f}",
            fontsize=12
        )
        
        # Redraw spectral lines for previously selected points
        for x, y, color in self.selected_points:
            self.ax[0].plot(x, y, "o", color=color, markersize=8)
            gt_spectrum = self.ground_truth[:, y, x]
            rc_spectrum = self.reconstructed[:, y, x]
            
            # Calculate pixel-specific fidelity
            pixel_fidelity = self.calculate_spectral_fidelity_for_pixel(x, y)
            
            line_gt, = self.ax[2].plot(self.wavelengths, gt_spectrum, color=color, 
                                       label=f"GT ({x},{y})")
            line_rc, = self.ax[2].plot(self.wavelengths, rc_spectrum, "--", color=color, 
                                      label=f"RC ({x},{y}) - Fid: {pixel_fidelity:.4f}")
            self.lines.append((line_gt, line_rc))
        
        if self.lines:
            self.ax[2].legend()
        
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events for spectral point selection."""
        if event.inaxes == self.ax[0]:  # Ensure click is on the ground truth plot
            x, y = int(event.xdata), int(event.ydata)
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            
            self.selected_points.append((x, y, color))
            
            # Mark the selected point
            self.ax[0].plot(x, y, "o", color=color, markersize=8)
            
            # Extract spectra
            gt_spectrum = self.ground_truth[:, y, x]
            rc_spectrum = self.reconstructed[:, y, x]
            
            # Calculate pixel-specific fidelity
            pixel_fidelity = self.calculate_spectral_fidelity_for_pixel(x, y)
            pixel_MAE = self.calculate_MAE_for_pixel(x,y)
            
            # Plot spectra with fidelity information
            line_gt, = self.ax[2].plot(self.wavelengths, gt_spectrum, color=color, 
                                       label=f"GT ({x},{y})")
            line_rc, = self.ax[2].plot(self.wavelengths, rc_spectrum, "--", color=color, 
                                      label=f"RC ({x},{y}) - MAE: {pixel_MAE :.4f}")
            self.lines.append((line_gt, line_rc))
            
            # Update legend
            self.ax[2].legend()
            self.fig.canvas.draw()
    
    def clear_spectra(self, event):
        """Clear all selected spectral points."""
        for line_gt, line_rc in self.lines:
            line_gt.remove()
            line_rc.remove()
        self.lines = []
        self.selected_points = []
        self.update_plot()
    
    def load_next_image(self, event):
        """Load the next image in the sequence."""
        self.current_index = (self.current_index + 1) % self.num_images
        try:
            self.load_images()
            self.clear_spectra(None)  # Clear any existing spectra
            self.update_plot()
        except Exception as e:
            print(f"Error loading next image: {e}")
            self.load_next_image(None)  # Try the next one

# Main execution

import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="Batch HSI comparison and metrics export.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--num_images', type=int, default=161, help='Number of images to process')
    parser.add_argument('--metrics_csv', type=str, default=None, help='Path to output metrics CSV (default: results_dir/metrics.csv)')
    args = parser.parse_args()

    image_directory = args.results_dir
    NUM_IMAGES = args.num_images
    metrics_csv = args.metrics_csv or os.path.join(image_directory, 'metrics.csv')

    analyzer = HyperspectralAnalyzer(image_directory, NUM_IMAGES)
    metrics = analyzer.calculate_all_metrics()

    # Write metrics to CSV
    with open(metrics_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics written to {metrics_csv}")

if __name__ == "__main__":
    main()