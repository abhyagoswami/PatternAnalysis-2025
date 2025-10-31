"""
train.py
Training loop for Improved 2D U-Net on Hip MRI data.
"""
import os
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from dataset import make_datasets

from dataset import make_dataloaders
from modules import ImprovedUNet

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

class BinaryDiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    This loss function measures the overlap between predicted and ground truth masks.
    """
    def __init__(self, threshold=1e-6):
        """
        Parameters:
            - threshold (float): threshold to avoid division by zero.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        """
        Computes the Dice Loss between logits and targets.
        Parameters:
            - logits (torch.Tensor): predicted logits from the model: shape (N, 1, H, W) raw.
            - targets (torch.Tensor): ground truth binary masks: shape (N, 1, H, W) 0/1 binary.
        Returns:
            - 1 - dice_coeff (torch.Tensor): Dice Loss value.
        """
        probabilities = torch.sigmoid(logits) # Convert logits to probabilities
        probabilities = probabilities.reshape(-1) # Flatten
        targets = targets.reshape(-1) # Flatten also for compatibility
        intersection = (probabilities * targets).sum() # Find where they overlap

        # Dice coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
        dice_coeff = (2 * intersection + self.threshold) / (probabilities.sum() + targets.sum() + self.threshold)
        return 1 - dice_coeff

def combined_loss(outputs: dict, targets: torch.Tensor, bce_weight: float, ds_weights=None):
    """
    Combines Binary Cross Entropy Loss and Dice Loss for better segmentation performance.
    If deep supervision is present, compute loss on all heads.
    Parameters:
        - outputs (dict): dictionary of model outputs at different scales.
        - targets (torch.Tensor): ground truth binary masks.
        - bce_weight (float): weight for the BCE loss component.
        - ds_weights (tuple of float): weights for deep supervision outputs. This is defaulted later when None.
    Returns:
        - total_loss (torch.Tensor): combined loss value.
    """
    bce = nn.BCEWithLogitsLoss()
    dice = BinaryDiceLoss()

    # Accept dict of outputs from ImprovedUNet ({"main":..., "aux2":..., "aux3":...}) or a single tensor
    if isinstance(outputs, dict):
        heads = [outputs["main"]]
        if "aux2" in outputs: # If deep supervision is used, include those losses too
            heads.append(outputs["aux2"])
        if "aux3" in outputs:
            heads.append(outputs["aux3"])
    elif isinstance(outputs, (list, tuple)):
        heads = list(outputs)
    else:
        heads = [outputs]

    # Default deep supervision weights are: main=1.0, aux2=0.5, aux3=0.25 (trim to number of heads)
    if ds_weights is None:
        ds_weights = (1.0, 0.5, 0.25)
    weights = ds_weights[:len(heads)] # Trim to number of outputs if we don't have all heads (aux2, aux3)

    sum_weights = 1.0 # If no deep supervision, just use 1.0
    if len(weights) > 0: 
        sum_weights = float(sum(weights))

    total_loss = 0.0
    i = 0
    while i < len(heads):
        head = heads[i]
        weight = weights[i]

        bce_loss = bce(head, targets)
        dice_loss = dice(head, targets)

        # Blend losses with the chosen weight
        mixed_loss = (bce_weight * bce_loss) + ((1.0 - bce_weight) * dice_loss)
        # Apply the deep supervision weight for this head from the tuple
        weighted_loss = weight * mixed_loss

        total_loss += weighted_loss
        i += 1
    
    return total_loss / sum_weights

@torch.no_grad()
def dice_coefficient(logits, targets, epsilon=1e-6):
    """
    Computes the Dice Coefficient on the MAIN head only between predicted logits and ground truth masks.
    Parameters:
        - logits (torch.Tensor): predicted logits from the model: shape (N, 1, H, W) raw.
        - targets (torch.Tensor): ground truth binary masks: shape (N, 1, H, W) 0/1 binary.
    Returns:
        - dice_coeff (float): Dice Coefficient value.
    """
    probabilities = torch.sigmoid(logits) # Convert logits to probabilities
    predictions = (probabilities > 0.5).float() # Check if each probability is > 0.5
    intersection = (predictions * targets).sum() # Find where predictions and targets overlap

    denominator = predictions.sum() + targets.sum() # Sum of predictions and targets for loss
    # Dice coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return dice.item()

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, bce_weight=0.5):
    """
    Trains a single epoch of the model.
    Parameters:
        - model (nn.Module): the ImprovedUNet model to train.
        - dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        - optimizer (torch.optim.Optimizer): optimizer for updating model weights.
        - device (torch.device): device to run the training on (CPU/GPU).
        - bce_weight (float): weight for the BCE loss component in combined loss. Default is 0.5.
    """
    model.train()

    running_loss = 0.0
    num_batches = 0 
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad() # Zero the gradients

        outputs = model(images) # Forward pass
        loss = combined_loss(outputs, masks, bce_weight) # Compute combined loss

        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        running_loss += float(loss.item()) # Accumulate loss
        num_batches += 1

    result = running_loss / max(1, num_batches) # Average loss over batches
    return result

def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device, bce_weight=0.5):
    """
    Validates the model on the validation dataset for a single epoch.
    Parameters:
        - model (nn.Module): the ImprovedUNet model to validate.
        - dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        - device (torch.device): device to run the validation on (CPU/GPU).
        - bce_weight (float): weight for the BCE loss component in combined loss. Default is 0.5.
    """
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images) # Forward pass
        loss = combined_loss(outputs, masks, bce_weight) # Compute combined loss

        
        if isinstance(outputs, dict): # If outputs is a dict from ImprovedUNet, then get MAIN head
            logits_main = outputs["main"]
        else: # Otherwise, assume outputs is the MAIN head directly
            logits_main = outputs

        # Dice on MAIN head only
        dice_coeff = dice_coefficient(logits_main, masks)
        running_loss += float(loss.item())
        running_dice += float(dice_coeff)
        num_batches += 1
    
    avg_loss = running_loss / max(1, num_batches) # Average loss over batches
    avg_dice = running_dice / max(1, num_batches) # Average dice over batches
    return avg_loss, avg_dice

@torch.no_grad()
def test_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device, bce_weight=0.5):
    """
    Tests the model for a single epoch.
    Parameters:
        - model (nn.Module): the ImprovedUNet model to test.
        - dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        - device (torch.device): device to run the testing on (CPU/GPU).
        - bce_weight (float): weight for the BCE loss component in combined loss. Default is 0.5.
    """
    return validate_epoch(model, dataloader, device, bce_weight=bce_weight)

"""
Plotting losses and metrics.
"""
import matplotlib.pyplot as plt

def plot(train_losses, val_losses, val_dices):
    """
    Plots training and validation losses and validation Dice coefficients over epochs.
    Parameters:
        - train_losses (list of float): training losses per epoch.
        - val_losses (list of float): validation losses per epoch.
        - val_dices (list of float): validation Dice coefficients per epoch.
    """
    # First we plot the losses 
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("losses.png")
    plt.close()

    # Now we plot the validation Dice coefficients
    plt.figure(figsize=(8,4))
    plt.plot(val_dices, label="validation dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dice.png")
    plt.close()
    print(f"Saved plots to losses.png and dice.png")


"""
This main function is AI generated (ChatGPT-5).
"""
if __name__ == "__main__":
    # --- config ---
    ROOT = os.path.expanduser("~/Desktop/HipMRI_2D")
    PROSTATE_ID = 1
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-3
    BCE_WEIGHT = 0.5
    DEEP_SUP = True

    device = pick_device()

    # data
    train_loader, val_loader, test_loader = make_dataloaders(
        root=ROOT, prostate_id=PROSTATE_ID, batch_size=BATCH_SIZE, num_workers=2, shuffle_train=True
    )

    # model + optim
    model = ImprovedUNet(in_channels=1, out_channels=1, base_channels=32,
                         dropout_prob=0.2, deep_supervision=DEEP_SUP).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_val_dice = 0.0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, bce_weight=BCE_WEIGHT)
        va_loss, va_dice = validate_epoch(model, val_loader, device, bce_weight=BCE_WEIGHT)

        train_losses.append(tr_loss); val_losses.append(va_loss); val_dices.append(va_dice)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), "improved_unet_best.pt")
            print(f"  ↳ Saved new best model (val_dice={best_val_dice:.4f})")
        
        if va_dice >= 0.98:
            print("Validation Dice reached 0.98, stopping training early.")
            break

    # final test
    test_loss, test_dice = test_epoch(model, test_loader, device, bce_weight=BCE_WEIGHT)
    print(f"TEST | loss={test_loss:.4f} | dice={test_dice:.4f}")
    torch.save(model.state_dict(), "improved_unet_last.pt")
    print("Saved final model to improved_unet_last.pt")

    # plots required by rubric
    plot(train_losses, val_losses, val_dices)


