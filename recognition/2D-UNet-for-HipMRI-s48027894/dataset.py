"""
dataset.py for HipMRI 2D slices loader.
Expected on-disk layout is (all containing .nii.gz file types):
    HipMRI_2D
    ├─ keras_slices_train/
    ├─ keras_slices_validate/
    ├─ keras_slices_test/
    ├─ keras_slices_seg_train/
    ├─ keras_slices_seg_validate/
    └─ keras_slices_seg_test/

"""

import os, glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

"""
The Appendix B NIfTI parsing code is given below.
"""

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

class _Utils: 
    pass
utils = _Utils()
utils.to_channels = to_channels

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory & usage.
    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    if categorical:
        first_case = to_channels (first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
         return images
    

"""
Project dataset is a simple PyTorch dataset for HipMRI 2D slices.
Here are the helper functions used within.
"""

def _read_nii2d(path: str) -> np.ndarray:
    """
    Load a single 2D nifti HipMRI_2D file and return it as a numpy array.
    Argument is file path as a string.
    """
    # Load the file from disk
    arr = nib.load(path).get_fdata(caching='unchanged')

    # The HipMRI slices are intended to be 2D, so we check
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0] # If incorrect shape, drop the last dimension

    # Use float32 for consistency with PyTorch
    return arr.astype(np.float32)

def _get_key(path: str) -> Tuple:
    """
    Create a key for filenames for easy matching.
    Parameters:
        - p: path stringxw
    Returns:
        A tuple of 3 integers: (case_id, week, slice).
    """
    # Get the file name without extension
    base = os.path.basename(path)
    remove_nii = base.split(".nii") # Remove .nii or .nii.gz
    base = remove_nii[0]
    parts = base.split("_") # Train/test start with "case" part and segmentations start with "seg"
    # print(parts)

    key = []
    for part in parts:
        if part.isdigit(): # Keep only the numeric pieces
            key.append(int(part))

    return tuple(key)

def _sort(path: str) -> List[str]:
    """
    Return all .nii / .nii.gz files in a folder, sorted by our numeric key.
    Parameters:
        - path: directory path for file folders (no folder should be in here, only .nii.gz files).
    Returns:
        - a list of sorted file paths that were found in the folder.
    """
    files = []
    for name in os.listdir(path):
        files.append(os.path.join(path, name))
    
    files.sort(key=_get_key) # Sort by numeric key
    return files

def _zscore(x: np.ndarray) -> np.ndarray:
    """
    Perform per-slice standardisation: (x - mean) / stddev
    This keeps range consistent across slices and helps stability during training.
    
    Parameters:
        - x: input numpy array (H, W) or (H, W, C) for multi-channel.
    Returns:
        - z-scored numpy array of the same shape as x.
    """
    mean, std = float(x.mean()), float(x.std())
    # We want to avoid zero-division error
    if std < 0.00001:
        return x * 0.0
    return (x - mean) / std

"""
PyTorch dataset using helpers.
"""
class HipMRI2DSlices(Dataset):
    """
    Build the HipMRI 2D prostate segmentation dataset for PyTorch.

    The nth file in the training/validation/test image folders corresponds to the nth file
    in the segmentation folders, sorted by numeric keys extracted from filenames using _sort.
    """
    def __init__(self, root: str, split: str, prostate_id: int):
        """
        Parameters:
            - root: path to the main HipMRI_2D folder containing the data subfolders.
            - split: one of "train", "validate", or "test" to select the dataset split.
            - prostate_id: integer label in the segmentation masks that corresponds to the prostate.
        """
        self.root = root
        self.split = split
        self.prostate_id = int(prostate_id)

        # Folders for this split
        images_path = os.path.join(root, f"keras_slices_{split}") # e.g., keras_slices_train
        masks_path = os.path.join(root, f"keras_slices_seg_{split}") # e.g., keras_slices_seg_train

        # Get sorted file lists
        self.images = _sort(images_path)
        self.masks = _sort(masks_path)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset split.
        """
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image and mask tensors for the given index.

        Parameters:
            - index: index of the sample to retrieve.
        Returns:
            - A tuple of (image_tensor, mask_tensor).
              image_tensor: shape (1, H, W), float32
              mask_tensor: shape (1, H, W), float32, binary mask for prostate_id
        """
        image_path = self.images[index]
        mask_path = self.masks[index]

        # (H, W) float32 arrays
        image = _read_nii2d(image_path)
        mask = _read_nii2d(mask_path)   

        # Normalise image per-slice
        image = _zscore(image)
        # Do element-wise comparison to get binary mask for prostate_id
        binary_mask = (mask == float(self.prostate_id)).astype(np.float32)

        # To tensors [1, H, W]
        x = torch.from_numpy(image).unsqueeze(0).float()
        y = torch.from_numpy(binary_mask).unsqueeze(0).float()

        return x, y

    

def make_datasets(root: str, prostate_id: int) -> Tuple[HipMRI2DSlices, HipMRI2DSlices, HipMRI2DSlices]:
    """
    Create PyTorch datasets for training, validation, and testing.

    Parameters:
        - root: path to the main HipMRI_2D folder containing the data subfolders.
        - prostate_id: integer label in the segmentation masks that corresponds to the prostate.
    Returns:
        - A tuple of (train_dataset, validate_dataset, test_dataset).
    """
    train_dataset = HipMRI2DSlices(root, "train", prostate_id)
    validate_dataset = HipMRI2DSlices(root, "validate", prostate_id)
    test_dataset = HipMRI2DSlices(root, "test", prostate_id)

    return train_dataset, validate_dataset, test_dataset

def make_dataloaders(root: str, prostate_id: int, batch_size: int, num_workers: int = 0, shuffle_train: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch dataloaders for training, validation, and testing.

    Parameters:
        - root: path to the main HipMRI_2D folder containing the data subfolders.
        - prostate_id: integer label in the segmentation masks that corresponds to the prostate.
        - batch_size: number of samples per batch.
        - num_workers: number of subprocesses to use for data loading.
        - shuffle_train: whether to shuffle the training data.
    Returns:
        - A tuple of (train_dataloader, validate_dataloader, test_dataloader).
    """
    train_dataset, validate_dataset, test_dataset = make_datasets(root, prostate_id)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, validate_loader, test_loader

# Testing code (Microsoft copilot was used to help write this entirely)
# root = os.path.expanduser("~/Desktop/HipMRI_2D")
# prostate_id = 1

# train, validate, test = make_datasets(root, prostate_id)
# print(f"[counts] train={len(train)}  val={len(validate)}  test={len(test)}")

# # Check one sample
# if len(validate):
#     x, y = validate[0]
#     print("sample shapes:", tuple(x.shape), tuple(y.shape)) # expect (1,H,W)

# # Build loaders (batch_size=1 is safest if H,W vary)
# train_loader, val_loader, test_loader = make_dataloaders(root, prostate_id, batch_size=4, num_workers=0)
# xb, yb = next(iter(train_loader))
# print("batch shapes:", tuple(xb.shape), tuple(yb.shape))  # expect (B,1,H,W)


