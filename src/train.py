"""
train.py — Train CLIP ViT-B/16 + linear head on iWildCam (CE baseline).

This implements the `--algorithm=ce` baseline that FLYP critiques.
Run this AFTER you've verified data.py and model.py both work.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_iwildcam_loaders
from model import CLIPClassifier


# ============================================================================
# Hyperparameters (matching FLYP paper Table 1 settings for CE baseline)
# ============================================================================

# These are taken from FLYP paper Section 4 / their codebase defaults.
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-5    # Important: CLIP fine-tuning uses very small lr
WEIGHT_DECAY = 0.2      # AdamW's L2 regularization strength
NUM_WORKERS = 4

# Where to save things
DATA_DIR = "./data"            # Where WILDS will download iWildCam (~100GB)
CHECKPOINT_DIR = "./results"   # Where to save model checkpoints


# ============================================================================
# Helper: evaluate model on a given DataLoader
# ============================================================================

def evaluate(model, loader, device):
    """
    Evaluate model on a DataLoader. Returns accuracy (% correct).
    """
    # TODO #1: switch model to evaluation mode (turns off dropout etc.)
    # Hint: one line, calling a method on `model`.
    model.eval()

    correct = 0
    total = 0

    # TODO #2: wrap the for-loop body in a `with torch.no_grad():` block.
    # This is correct usage of no_grad — eval doesn't need gradients.
    # Inside the loop:
    #   - move x and y to device: x = x.to(device); y = y.to(device)
    #   - forward: logits = model(x)
    #   - get predictions: preds = logits.argmax(dim=1)
    #   - count correct: correct += (preds == y).sum().item()
    #   - count total: total += y.size(0)

    with torch.no_grad():
        for x,y,metadata in loader:
            x=x.to(device)
            y=y.to(device)

            logits=model(x)
            preds=logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


# ============================================================================
# Main training loop
# ============================================================================

def train():
    # ----- Setup -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----- Data -----
    print("Loading data ...")
    loaders = get_iwildcam_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"  train batches: {len(loaders['train'])}")
    print(f"  id_val batches: {len(loaders['id_val'])}")
    print(f"  val (OOD) batches: {len(loaders['val'])}")

    # ----- Model -----
    print("Creating model ...")
    model = CLIPClassifier().to(device)

    # ----- Loss, optimizer, scheduler -----
    # TODO #3: create the loss function.
    # We use cross-entropy loss for multi-class classification.
    # Hint: nn.CrossEntropyLoss()
    # Store it in a variable called `criterion`
    criterion = nn.CrossEntropyLoss()

    # TODO #4: create the optimizer.
    # AdamW over ALL model parameters (model.parameters()).
    # lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # TODO #5: create the learning rate scheduler.
    # CosineAnnealingLR with T_max = total number of training steps
    # = EPOCHS * len(loaders['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loaders['train']))

    # ----- Training loop -----
    for epoch in range(EPOCHS):
        # TODO #6: switch model to training mode
        # Hint: model.train()
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{EPOCHS}")

        for x, y, metadata in progress_bar:
            x = x.to(device)
            y = y.to(device)

            # TODO #7: the 5-step training pattern
            # Step 1: zero gradients   (optimizer.zero_grad())
            # Step 2: forward pass     (logits = model(x))
            # Step 3: compute loss     (loss = criterion(logits, y))
            # Step 4: backward pass    (loss.backward())
            # Step 5: optimizer step   (optimizer.step())
            # Then also call scheduler.step() to advance the LR schedule.

            optimizer.zero_grad()               # Step 1
            logits = model(x)                   # Step 2
            loss = criterion(logits, y)         # Step 3
            loss.backward()                     # Step 4
            optimizer.step()                    # Step 5
            scheduler.step()                    # Step 6 (LR schedule step)

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            

        avg_loss = running_loss / len(loaders['train'])
        print(f"\nEpoch {epoch+1} avg train loss: {avg_loss:.4f}")

        # ----- Evaluate on id_val -----
        id_val_acc = evaluate(model, loaders['id_val'], device)
        print(f"Epoch {epoch+1} id_val accuracy: {id_val_acc:.2f}%")

        # ----- Save checkpoint -----
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ce_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("\nTraining done.")


if __name__ == "__main__":
    train()