"""
model.py — CLIP ViT-B/16 with a linear classification head for iWildCam.

The "CE baseline" architecture that FLYP critiques:
  - Frozen-pretrained CLIP ViT-B/16 image encoder (the "backbone")
  - Randomly initialized linear head mapping 512-dim features → 182-dim logits
  - Trained end-to-end with cross-entropy loss
"""

import torch
import torch.nn as nn
import open_clip


# ============================================================================
# Constants
# ============================================================================

# CLIP ViT-B/16 image encoder outputs a 512-dimensional feature vector.
CLIP_FEATURE_DIM = 512

# iWildCam has 182 classes (181 species + 1 "empty" class).
NUM_CLASSES = 182


# ============================================================================
# Model definition
# ============================================================================

class CLIPClassifier(nn.Module):
    """
    CLIP image encoder + linear classification head.

    This is the architecture used for the `--algorithm=ce` baseline.
    Both the backbone and the head are trainable (full fine-tuning).
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # ----------------------------------------------------------------
        # Load CLIP ViT-B/16 backbone using open_clip.
        # open_clip.create_model_and_transforms returns:
        #   (model, train_preprocess, val_preprocess)
        # We only need `model.visual` — the image encoder part of CLIP.
        # ----------------------------------------------------------------
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        # Keep ONLY the image encoder; we don't need the text encoder for CE baseline.
        self.backbone = clip_model.visual

        # ----------------------------------------------------------------
        # TODO #1: Create the linear classification head.
        # It should map CLIP_FEATURE_DIM (512) → num_classes (182).
        # Hint: use nn.Linear(in_features, out_features)
        # Store it in self.head
        # ----------------------------------------------------------------
        self.head = nn.Linear(CLIP_FEATURE_DIM,num_classes)

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: tensor of shape (batch_size, 3, 224, 224)

        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        # TODO #2: Pass images through the CLIP backbone to get features.
        # The call: features = self.backbone(images)
        # Note: features will have shape (batch_size, 512)
        
        features = self.backbone(images)


        # TODO #3: Pass features through the classification head to get logits.
        # The call: logits = self.head(features)
        # Then `return logits`
        logits = self.head(features)

        return logits



# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("model.py loaded successfully")
    print("Creating CLIPClassifier (this loads CLIP weights, takes ~30s) ...")

    model = CLIPClassifier()

    print(f"Backbone type: {type(model.backbone).__name__}")
    print(f"Head: {model.head}")

    # Quick shape check with random input
    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2 random images
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: torch.Size([2, {NUM_CLASSES}])")