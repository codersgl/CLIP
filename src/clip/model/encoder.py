from abc import ABC, abstractmethod
from typing import Optional

import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class Encoder(nn.Module, ABC):
    """Abstract base class for all CLIP encoders.

    Each encoder must implement:
    - A feature extraction backbone (defined by subclasses in __init__)
    - A forward method that returns projected and normalized features

    Common parts:
    - A linear projection layer that maps backbone outputs to a unified embedding space.
    - Optional LayerNorm (not used in original CLIP, but some variants may use it)

    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of the final embedding space.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Projection layer: maps backbone features to embedding space.
        # Note: The backbone output dimension (mid_dim) is unknown at this point,
        # so we delay creation. We'll initialize it later via _build_projection.
        self.projection: Optional[nn.Linear] = None

    def _build_projection(self, mid_dim: int) -> None:
        """Create the projection layer (called by subclasses after backbone is known)."""
        self.projection = nn.Linear(mid_dim, self.embed_dim)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass. Returns a tensor of shape (batch_size, embed_dim).
        Subclasses must implement this and call self.projection.
        """
        pass


class VisualEncoder(Encoder):
    """
    Visual encoder: uses a timm model as backbone, followed by projection.
    """

    def __init__(
        self,
        embed_dim: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        **kwargs,  # Additional arguments for timm.create_model
    ):
        # Call parent initializer (does not set mid_dim)
        super().__init__(embed_dim)

        # Load backbone, removing the classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            **kwargs,
        )

        # Get backbone output dimension (timm models usually have num_features)
        mid_dim = self.backbone.num_features
        assert isinstance(mid_dim, int), f"Expected int, got {type(mid_dim)}"
        self._build_projection(mid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)  # (batch, mid_dim)
        # Project
        if self.projection is None:
            raise RuntimeError("Projection layer not initialized.")
        projected = self.projection(features)  # (batch, embed_dim)
        return projected


class TextEncoder(Encoder):
    """
    Text encoder: uses a HuggingFace Transformer model as backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        model_name: str = "distilbert-base-uncased",
        pooler: str = "cls",  # Options: "cls" or "mean"
        **kwargs,
    ):
        super().__init__(embed_dim)

        # Load backbone model
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooler = pooler

        # Get backbone output dimension (usually hidden_size)
        mid_dim = self.backbone.config.hidden_size
        self._build_projection(mid_dim)

        assert self.projection is not None, "Projection layer should be initialized"
        # Optional tokenizer (if you need to process text inside, not used here)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Backbone forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling strategy
        if self.pooler == "cls":
            # Use [CLS] token's last layer output
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # Mean pooling (accounting for padding via attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            features = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        # Project
        if self.projection is None:
            raise RuntimeError("Projection layer not initialized.")
        projected = self.projection(features)
        return projected
