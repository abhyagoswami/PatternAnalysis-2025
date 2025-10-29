"""
dataset.py for HipMRI 2D slices loader.
This uses the Appendix B NIfTI parsing code.
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm