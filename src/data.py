"""
data.py — iWildCam dataset loader for FLYP baseline reproduction.

Loads the iWildCam dataset from WILDS, sets up image preprocessing
that matches CLIP ViT-B/16 expectations, and returns DataLoaders for
each split (train, id_val, val=OOD).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset


# ============================================================================
# Constants — these match CLIP ViT-B/16's preprocessing requirements
# ============================================================================

# CLIP was pretrained on images normalized with these mean/std values.
# Don't change these — they must match how the backbone was trained.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP ViT-B/16 expects 224×224 input images
IMAGE_SIZE = 224


# ============================================================================
# Image preprocessing pipeline
# ============================================================================

def get_transform():
    """
    Returns the image preprocessing pipeline.

    Steps:
        1. Resize the image so the shorter side is IMAGE_SIZE
        2. Center crop to IMAGE_SIZE × IMAGE_SIZE
        3. Convert PIL image to PyTorch tensor (also scales to [0,1])
        4. Normalize using CLIP_MEAN and CLIP_STD
    """
    # TODO #1: Use transforms.Compose to chain together the 4 steps above.
    # Hint: each step is a transforms.XXX(...) call.
    # The functions you'll need:
    #   - transforms.Resize(IMAGE_SIZE)
    #   - transforms.CenterCrop(IMAGE_SIZE)
    #   - transforms.ToTensor()
    #   - transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean= CLIP_MEAN,std=CLIP_STD)
    ])
    return transform


# ============================================================================
# Dataset loaders
# ============================================================================

def get_iwildcam_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Loads iWildCam from WILDS and returns DataLoaders for each split.

    Args:
        data_dir: path to the WILDS data folder (will download here if missing).
        batch_size: how many images per batch.
        num_workers: how many subprocesses for data loading.

    Returns:
        A dict with keys 'train', 'id_val', 'val' (=OOD val).
        Values are torch.utils.data.DataLoader objects.
    """
    # Step 1: Load the iWildCam dataset from WILDS.
    # The WILDS API: dataset = get_dataset(dataset='iwildcam', root_dir=data_dir, download=True)
    # TODO #2: write the line above (uncomment + use the right variable).
    dataset = get_dataset(dataset='iwildcam', root_dir=data_dir, download=True)

    # Step 2: Get the shared transform pipeline.
    # TODO #3: call get_transform() once and store it in a variable.
    transform = get_transform()

    # Step 3: For each split, get the WILDS subset and wrap it in a DataLoader.
    # The WILDS API for getting a split's subset:
    #     subset = dataset.get_subset('train', transform=transform)
    #
    # The DataLoader call looks like:
    #     loader = DataLoader(subset, batch_size=batch_size,
    #                         shuffle=is_train, num_workers=num_workers)
    # Note: shuffle=True ONLY for training. Eval splits should have shuffle=False.

    loaders = {}

    # TODO #4: Create the train loader.
    # Split name: 'train'
    # Shuffle: True
    # Store in loaders['train']
    train_subset = dataset.get_subset('train',transform=transform)
    train_loader = DataLoader(
        train_subset,
        batch_size = batch_size,
        shuffle = True,
        num_workers=num_workers
    )
    loaders['train'] = train_loader

    # TODO #5: Create the id_val loader.
    # Split name: 'id_val'
    # Shuffle: False
    # Store in loaders['id_val']
    id_val_subset = dataset.get_subset('id_val',transform=transform)
    id_val_loader =DataLoader(
        id_val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers

    )
    loaders['id_val']= id_val_loader

    # TODO #6: Create the OOD val loader.
    # Split name: 'val'  (in WILDS, 'val' means OOD val for iWildCam)
    # Shuffle: False
    # Store in loaders['val']
    val_subset = dataset.get_subset('val', transform=transform)
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    loaders['val'] = val_loader


    return loaders


# ============================================================================
# Sanity check — run this file directly to verify it works
# ============================================================================

if __name__ == "__main__":
    # When you run `python src/data.py` directly, this block executes.
    # Useful for quick sanity-checking the file before integrating with training.
    print("data.py loaded successfully")
    print("CLIP_MEAN:", CLIP_MEAN)
    print("IMAGE_SIZE:", IMAGE_SIZE)
    # Note: We won't actually run get_iwildcam_loaders() here because that
    # would trigger the 100GB download. We'll test that on Colab.