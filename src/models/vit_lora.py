from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.models.lora import LoRALinear


def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: int, dropout: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            _replace_linear_with_lora(child, rank, alpha, dropout)


class ViTLoRAClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 7,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)

        original_proj = self.backbone.conv_proj
        new_proj = nn.Conv2d(
            in_channels,
            original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=False,
        )

        with torch.no_grad():
            new_proj.weight[:, :3] = original_proj.weight
            if in_channels > 3:
                mean_weight = original_proj.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_proj.weight[:, c:c + 1] = mean_weight

        self.backbone.conv_proj = new_proj

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.conv_proj.weight.requires_grad = True
        _replace_linear_with_lora(self.backbone.encoder, rank=rank, alpha=alpha, dropout=dropout)

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total