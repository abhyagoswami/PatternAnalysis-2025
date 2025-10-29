"""
Helper/preview files for inspection and debugging only.
The local dataset used was in this format (all containing .nii.gz file types):
    HipMRI_2D
    ├─ keras_slices_train/
    ├─ keras_slices_validate/
    ├─ keras_slices_test/
    ├─ keras_slices_seg_train/
    ├─ keras_slices_seg_validate/
    └─ keras_slices_seg_test/

First code previews images and ensures they load properly.
"""

import os, glob, argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def read_nii2d(path: str) -> np.ndarray:
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

def preview_image(image_path: str, mask_path: str, prostate_id: int | None = None, alpha: float = 0.3) -> None:
    """
    Display a simple overlay of a 2D MRI slice with its mask in grayscale.

    Parameters:
        - image_path: string path to the image (nifti)
        - mask_path: string path to mask (nifti)
        - prostate_id: if provided, we only show where mask == prostate_id.
            if None, we overlay the full mask (all labels).
        - alpha: transperancy for overlay (0 = invisible, 1 = opaque).
    """
    # Load arrays from disk
    img = read_nii2d(image_path) # shape (H, W), floats
    mask = read_nii2d(mask_path)

    if prostate_id is None:
        # Show the entire mask (all labels). This is useful to inspect what's inside first.
        overlay = mask
        title = f"{os.path.basename(image_path)}  (all labels overlay)" # Title for graph
    else:
        # Binary overlay: 1 where mask equals the given label, else 0.
        overlay = (mask == float(prostate_id)).astype(np.float32)
        title = f"{os.path.basename(image_path)}  (prostate_id = {prostate_id})"

    # Make the figure
    figure, axis = plt.subplots(1, 1, figsize=(6, 6))
    axis.imshow(img, cmap="gray") # base MRI in grayscale
    axis.set_aspect('auto')
    show = np.ma.masked_where(overlay <= 0, overlay)
    axis.imshow(show, alpha=0.35, cmap='autumn', interpolation='nearest', vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")

    plt.show()

# preview_image(
#     image_path="recognition/2D-UNet-for-HipMRI-s48027894/HipMRI_2D/keras_slices_train/case_004_week_0_slice_0.nii.gz",
#     mask_path ="recognition/2D-UNet-for-HipMRI-s48027894/HipMRI_2D/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz",
#     prostate_id=1
# )

