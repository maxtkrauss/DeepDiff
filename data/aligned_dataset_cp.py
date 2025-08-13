import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + '/thorlabs')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + '/cubert')  # create a path '/path/to/data/trainB'

        self.dir_A = os.path.join(opt.dataroot, 'validation' + '/thorlabs')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'validation' + '/cubert')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
        index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor)       -- an image in the input domain
        B (tensor)       -- its corresponding image in the target domain
        A_paths (str)    -- image paths
        B_paths (str)    -- image paths
        """
        '''
        AB_path = self.AB_paths[index]
        base_name = os.path.basename(AB_path)
        AB_path = os.path.join(AB_path, base_name)
        
        base_name_2 = base_name.replace('_ms', '')
        A_path = os.path.join(AB_path, base_name_2 + '_RGB.bmp')
        A = Image.open(A_path).convert('L')
        
        # Load hyperspectral images
        B_images = []
        #B_paths = sorted([os.path.join(AB_path, fname) for fname in os.listdir(AB_path) if self.is_image_file(fname)])

        for i in range(1, 60):
            filename = f'{base_name}_{i:02d}.png'
            B_path = os.path.join(AB_path, filename)
            #print(f'b path: {B_path}')
            B_image = Image.open(B_path)  # Convert to grayscale

            # Check min, max, and bit depth
            B_array = np.array(B_image)
            #print(f'B_image {i} - min: {B_array.min()}, max: {B_array.max()}, dtype: {B_array.dtype}, shape: {B_array.shape}')

            B_images.append(B_image)
'''
        
        A_path = self.A_paths[index % self.A_size]
        index_B = index % self.B_size

        B_path = self.B_paths[index_B]
        #print(f'A path: {A_path}')
        #print(f'B path: {B_path}')

        A = io.imread(A_path) # Shape: (5, 660, 660)
        B = io.imread(B_path) # Shape: (106, 120, 120)

        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())

        A = A[:1, :, :] # Shape: (1, 660, 660)
        B = B[:64, :, :] # Shape: (60, 120, 120)

        B = np.pad(B, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0) # Pad to 60x128x128

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        A = A.unsqueeze(0)  # Add batch dimension
        A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False) # Resize to 120x120
        A = A.squeeze(0)  # Remove batch dimension

        #B = B.unsqueeze(0)  # Shape: (1, 60, 120, 120)
        #B = F.interpolate(B, size=(128, 128), mode='bilinear', align_corners=False)
        #B = B.squeeze(0)  # Shape: (60, 256, 2656)

        #print(f'A shape: {A.shape}')
        #print(f'B shape: {B.shape}')

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
