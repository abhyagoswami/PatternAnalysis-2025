"""
modules.py
Improved U-Net (2D) for binary prostate segmentation.
The model offers the following upgrades from the original U-Net architecture:
- Pre-activation residual blocks
- InstanceNorm
- Dropout
- Optional deep supervision

These improvements were based on the lecture slides, from the listed paper [1] on task sheet and the provided
code demo on U-Net segmentations (blackboard).
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A pre-activation residual block with two convolutional layers.
    Each convolution is preceded by InstanceNorm and Leaky ReLU activation.
    Dropout is applied after the first convolution if specified by a probability.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.2):
        """
        Initialise the residual block.
        Parameters:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - dropout_prob: probability of dropout after the first convolution (default 0.2)
        """
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True) # Normalise each channel so that mean = 0 and var = 1
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True) # We choose LeakyReLU to avoid dead neurons
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_prob)

        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True) # Now normalise the output channels
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # If the input and output channels differ, we need a convolution to match dimensions for the skip connection
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        Parameters:
            - x: input tensor of shape (N, C_in, H, W)
        Returns:
            - output tensor of shape (N, C_out, H, W)
        """
        identity = x # Save input for the skip connection if needed

        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        if self.projection is not None:
            identity = self.projection(identity) # Match dimensions if needed
        return out + identity # Add skip connection
    
class UpBlock(nn.Module):
    """
    A simple upsampling block using bilinear upsampling followed by a residual block.
    It performs an upsample (bilinear) followed by concatenation (skip connection), 
    then the pre-activation residual block.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_prob: float = 0.2):
        super().__init__()
        # We use a scale factor of 2 because we want to double the spatial dimensions
        # The 'bilinear' mode provides smooth upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # After upsampling, we concatenate the skip connection channels
        self.fuse = ResidualBlock(in_channels + skip_channels, out_channels, dropout_prob)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        Parameters:
            - x: input tensor of shape (N, C_in, H, W)
            - skip: skip connection tensor of shape (N, C_skip, H*2, W*2)
        Returns:
            - output tensor of shape (N, C_out, H*2, W*2)
        """
        x = self.upsample(x) # Upsample the input
        # If spatial sizes differ due by 1 pixel, we pad x to match the skip's size
        if x.shape[2:] != skip.shape[2:]:
            # Center crop the skip connection and pad if necessary
            diff_y = skip.size(-2) - x.size(-2) # Difference in height
            diff_x = skip.size(-1) - x.size(-1) # Difference in width
            # Pad with differences divided equally on both sides
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x, skip], dim=1) # Concatenate along channel dimension
        return self.fuse(x)
        
class ImprovedUNet(nn.Module):
    """
    An improved U-Net encoder-decoder architecture for binary prostate segmentation consisting of:
    - Pre-activation residual blocks
    - InstanceNorm
    - Dropout
    - Optional deep supervision heads at each decoder level out of 2

    This outputs logits for binary segmentation.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, dropout_prob: float = 0.2, deep_supervision: bool = True):
        """
        Initialise the improved U-Net model.
        Parameters:
            - in_channels: number of input channels (default 1 for grayscale MRI)
            - out_channels: number of output channels (default 1 for binary segmentation)
            - base_channels: number of channels in the first layer (default 32)
            - dropout_prob: dropout probability in residual blocks (default 0.2)
            - deep_supervision: whether to use deep supervision heads (default True)
        """
        super().__init__()
        self.deep_supervision = deep_supervision # Store deep supervision flag (bool)

        # First we define the encoder path
        self.enc1 = ResidualBlock(in_channels, base_channels, dropout_prob) # from in_channels = 1 to 32
        self.pool1 = nn.MaxPool2d(2) # 2x2 max pooling for downsampling, performs H/2
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, dropout_prob) # 32 to 64
        self.pool2 = nn.MaxPool2d(2) # H/4
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, dropout_prob) # 64 to 128
        self.pool3 = nn.MaxPool2d(2) # H/8

        # Bottleneck of the U-Net is another residual block which further processes the features
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8, dropout_prob) # 128 to 256

        # Now we define the decoder path with upsampling blocks
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout_prob) # 256 + 128 to 128
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout_prob) # 128 + 64 to 64
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels, dropout_prob) # 64 + 32 to 32

        # Main output head
        self.out_head = nn.Conv2d(base_channels, out_channels, kernel_size=1) # from 32 to 1

        # Heads for deep supervision outputs
        # This allows the model to make predictions at multiple scales, allowing earlier layers to learn useful features
        if self.deep_supervision == True:
            self.head2 = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1) # from 64 to 1
            self.head3 = nn.Conv2d(base_channels * 4, out_channels, kernel_size=1) # from 128 to 1

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the improved U-Net.
        Parameters:
            - x: input tensor of shape (N, C_in, H, W)
        """
        # Encoder first
        enc1 = self.enc1(x) # (N, 32, H, W)
        enc2 = self.enc2(self.pool1(enc1)) # (N, 64, H/2, W/2)
        enc3 = self.enc3(self.pool2(enc2)) # (N, 128, H/4, W/4)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3)) # (N, 256, H/8, W/8)

        # Decoder with skip connections
        dec3 = self.up3(bottleneck, enc3) # (N, 128, H/4, W/4)
        dec2 = self.up2(dec3, enc2) # (N, 64, H/2, W/2)
        dec1 = self.up1(dec2, enc1) # (N, 32, H, W)

        main = self.out_head(dec1) # Main output head (N, 1, H, W)

        if self.deep_supervision == False:
            return {"main": main} # Return only the main output
        else:
            # Perform deep supervision by generating outputs at multiple scales
            aux2 = self.head2(dec2) # (N, 1, H/2, W/2)
            aux3 = self.head3(dec3) # (N, 1, H/4, W/4)
            # Upsample auxiliary outputs to match main output size
            aux2 = F.interpolate(aux2, size=main.shape[-2:], mode="bilinear", align_corners=False)
            aux3 = F.interpolate(aux3, size=main.shape[-2:], mode="bilinear", align_corners=False)
            return {"main": main, "aux2": aux2, "aux3": aux3} # Return all outputs
