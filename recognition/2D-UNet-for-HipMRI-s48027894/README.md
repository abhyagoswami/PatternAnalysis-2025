# Improved 2D U-Net for Hip MRI Prostate Segmentation
## 1 Overview
This project tackles binary prostate segmentation from 2D Hip MRI slices in the nifti format. 
It emplements an **improved U-Net** that adds pre activation residual blocks, InstanceNorm, Dropout in 20% cases, and optional deep supervision to the classic U-Net encoder decoder structure.

The goal is to get accuracye pizel-wise segmentation masks for the prostate which are evaluated by Dice similarity score.
For visualisation, losses.png plots train/validation loss and dice.png plots validation dice score.

## 2 Algorithm
The archiecture is as follows (with more comprehensive commenting in the code itself):
- **Encoder-decoder U-Net** with three down/upsampling stages and a bottleneck.
- **Residual pre-activation blocks** with InstanceNorm then LeakyReLU then Conv2D.
- **Optional dropout** between convolutions and a 1 $\times$ 1 projection on skip if channels change.
- **InstanceNorm** instead of BatchNorm for stability at small batch sizes.
- **Deep supervision** with auxiliary 1 $\times$ 1 heads from decoder stages that are upsampled and combined into the loss to improve gradient flow and regularisation.

The loss metrics are:
- **Combined loss** with $ \lambda \times BCE + (1 - \lambda) \times Dice $ where $\lambda$ defaults to $0.5$.
- **Deep supervision weights** are set at default for main head at 1.0, auxiliary head 2 at 0.5, and auxiliary head 3 at 0.25 (trimmed).
- **Dice** coefficient is used on the main head.

Preprocessing was also done with the following methods:
- NIFTI parsing via provided code from Appendix B.
- Standardise each slice via Z-score with mean = 0 and std = 1.0: $z = \frac{(x − \mu)}{\sigma} $.


## 3 Data Layout 
The expected disk data layout is:

HipMRI_2D/
├── keras_slices_train/ 
├── keras_slices_validate/
├── keras_slices_test/
├── keras_slices_seg_train/
├── keras_slices_seg_validate/
└── keras_slices_seg_test/

where HipMRI_2D should be placed inside the recognition/2D-UNet-for-HipMRI-s48027894
We use the provided splits from Rangpur to avoid leakage and ensure fair assessment.
The validation set tracks model selection/early stopping for epochs once validation dice is >98%, and the final test set is used only **once** to report performance. This prevents any bleeding and false scores, and mirrors standard practice.

## 4 Files
- modules.py - model components (ResidualBlock, UpBlock, ImprovedUNet).
- dataset.py - Dataset + DataLoader generation, NIFTI parsing, z-score preprocessing.
- train.py - train/validate/test loops, plotting (losses.png, dice.png), model saving.
- predict.py — example inference & overlay creation (see usage below).
- README.md - this report.
- utils.py - optional helper/tester file to ensure files parse correctly (this is not needed).

## 5 Dependencies + Extras
python >= 3.10
torch >= 2.2
torchvision >= 0.17 (if installed, but not required here). Other versions may cause conflict.
nibabel >= 5.2
numpy >= 1.26
matplotlib >= 3.8

For speed on NVIDIA GPUs, we use cudnn.benchmark=True.
During training, the device is chosen based on the best out of 3: CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

## 6 Running Code
1. Put the dataset at ~/Desktop/HipMRI_2D/ (or change ROOT in train.py).
2. run: python train.py
3. This outputs the best model, losses.png and dices.png.
