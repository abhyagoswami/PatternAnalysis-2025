"""
predict.py
Predict script for Improved 2D U-Net on Hip MRI data.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import make_datasets
from modules import ImprovedUNet

ROOT = os.path.expanduser("~/Desktop/HipMRI_2D")  # Change this to another path if needed
PROSTATE_ID = 1
WEIGHTS = "improved_unet_best.pt" # Path to trained weights (modify if needed)
N_SAMPLES = 6 # Number of test samples to predict and save overlays for
THRESHOLD = 0.5  # Probability threshold for binary mask

def pick_device():
    """
    Picks the most optimal device out of those currently available.
    """
    if torch.cuda.is_available():
        print("Using external GPU (CUDA)")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        print("No GPU found, using CPU")
        return torch.device("cpu")
    
@torch.no_grad()
def dice_binary(prediction_binary: torch.Tensor, target_binary: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Perform Dice on binary prediction and target tensors.
    Parameters:
        - prediction_binary: Binary prediction tensor, shape (1, H, W) or (N,1,H,W).
        - target_binary: Binary target tensor, shape (1, H, W) or (N,1,H,W).
        - eps: Small epsilon value to avoid division by zero.
    Returns:
        - dice_score: Dice score as a float.
    """
    # Flatten over all dims
    prediction = prediction_binary.reshape(-1).float() # (H*W,) or (N*H*W,)
    target = target_binary.reshape(-1).float()
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()
    # Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return dice_score.item()

def main(args):
    # Set device
    device = pick_device()

    # Create dataset
    train_data, validation_data, test_data = make_datasets(args.data_dir, args.img_size)
    datasets = {"train": train_data, "validate": validation_data, "test": test_data}

    # Load model
    model = ImprovedUNet(in_channels=1, out_channels=1, deep_supervision=True).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Pick some sample indices

"""
AI (GPT-5) was used to generate functions including: save_overlay, pick_indices.
"""

def save_overlay(image: np.ndarray, ground_truth: np.ndarray, 
                 prediction: np.ndarray, probability_map: np.ndarray, 
                 save_path: str, dice_score: float):
    """
    Save an overlay of 4 images with:
    - Original image
    - Ground truth mask
    - Prediction mask overlay
    - Heat map

    All inputs are simple NumPy arrays with shape (H, W).
    AI GENERATED.
    """
    figure, axis = plt.subplots(1, 4, figsize=(14, 3.5))

    # Panel 1: show the raw MRI slice in grayscale
    axis[0].imshow(image, cmap="gray")
    axis[0].set_title("Original Image")
    axis[0].axis("off")

    # Panel 2: overlay the ground-truth mask (orange) on top of the image
    axis[1].imshow(image, cmap="gray")
    axis[1].imshow(np.ma.masked_where(ground_truth <= 0, ground_truth), alpha=0.35, cmap="autumn")
    axis[1].set_title("Ground Truth Mask")
    axis[1].axis("off")

    # Panel 3: overlay the predicted binary mask (blue/green) on top of the image
    axis[2].imshow(image, cmap="gray")
    axis[2].imshow(np.ma.masked_where(prediction <= 0, prediction), alpha=0.35, cmap="winter")
    axis[2].set_title(f"Pred overlay (Dice {dice_score:.3f})")
    axis[2].axis("off")

    # Panel 4: show the raw probabilities as a heatmap
    image = axis[3].imshow(probability_map, cmap="viridis")
    axis[3].set_title("Probability Map")
    axis[3].axis("off")
    figure.colorbar(image, ax=axis[3], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)

def pick_indices(num_samples: int, n: int):
    """
    Pick n evenly spaced indices from a dataset of length num_samples.
    This is deterministic (same picks each run), and spreads picks evenly instead of clustering at start.
    Parameters:
        - num_samples: Total number of samples in the dataset.
        - n: Number of indices to pick.
    AI GENERATED.
    """
    n = max(1, min(n, num_samples)) # Make n in [1, num_samples]
    step = max(1, num_samples // n)
    return list(range(0, num_samples, step))[:n]

def main():
    # 1. Pick device
    device = pick_device()

    # 2. Load data
    train_data, validation_data, test_data = make_datasets(ROOT, PROSTATE_ID)

    # 3. Build model and load weights
    # This should match training configuration
    model = ImprovedUNet(in_channels=1, out_channels=1, base_channels=32, dropout_prob=0.2, deep_supervision=True).to(device)

    print(f"Loading weights: {WEIGHTS}")
    state = torch.load(WEIGHTS, map_location=device) # During training, we saved a model.state_dict()
    model.load_state_dict(state)
    model.eval()

    # 4. Pick samples to predict
    indices = pick_indices(len(test_data), N_SAMPLES)
    print(f"Predicting {len(indices)} samples: {indices}")

    # 5. Predict and save overlays
    dice_scores = []
    with torch.no_grad():
        for i, index in enumerate(indices, start=1):
            image, target = test_data[index]  # image: (1,H,W), binary mask: (1,H,W)
            # Prepare batch
            image_batch = image.unsqueeze(0).to(device) # (1,1,H,W)

            # Forward pass
            out = model(image_batch)
            logits = out
            if isinstance(out, dict): # If deep supervision is enabled, use the "main" output
                logits = out["main"]

            # Convert logits to probabilities in [0,1] with sigmoid
            probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            prediction_binary = (probabilities >= THRESHOLD).astype(np.uint8) # Make prediction binary mask

            # Compute Dice score
            dice = dice_binary(torch.from_numpy(prediction_binary).unsqueeze(0), target)
            dice_scores.append(dice)

            # Save figure
            image_np = image[0].detach().cpu().numpy()
            target_np = target[0].detach().cpu().numpy()
            out_path = f"test_{index:04d}.png" # Save in current directory
            save_overlay(image_np, target_np, prediction_binary, probabilities, out_path, dice)
            print(f"[{i}/{len(indices)}] index = {index}  Dice = {dice:.4f}, saved to {out_path}")

    mean_dice = 0.0
    if dice_scores: # Avoid division by zero
        mean_dice = float(np.mean(dice_scores))
    else:
        mean_dice = 0.0

    print(f"\nMean Dice over {len(indices)} test samples: {mean_dice:.4f}")

if __name__ == "__main__":
    main()