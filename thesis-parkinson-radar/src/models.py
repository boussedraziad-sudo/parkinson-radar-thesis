"""
Model architectures.

`SmallCNN` is the from-scratch baseline — small enough to train on a Mac CPU
in a few minutes per fold. `resnet18_finetune` adapts torchvision's ResNet-18
to accept N input channels and a binary head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """3-block CNN with global-average pooling. ~50k parameters."""

    def __init__(self, in_channels: int = 2, num_classes: int = 2, hidden: int = 64):
        super().__init__()
        c1, c2, c3 = hidden // 4, hidden // 2, hidden
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def resnet18_finetune(
    in_channels: int = 2,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_until: str | None = "layer3",
) -> nn.Module:
    """
    ResNet-18 adapted to `in_channels` input and a binary head.

    If pretrained weights are loaded and in_channels != 3, conv1's weights are
    averaged across the original 3 ImageNet channels and tiled. `freeze_until`
    freezes all layers up to and including the named block (use None to
    fine-tune everything; "layer3" leaves only layer4 + fc trainable).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)

    if in_channels != 3:
        old = net.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                avg = old.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                new.weight.copy_(avg.repeat(1, in_channels, 1, 1))
        net.conv1 = new

    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if freeze_until:
        block_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        if freeze_until in block_order:
            idx = block_order.index(freeze_until)
            for name in block_order[: idx + 1]:
                mod = getattr(net, name)
                for p in mod.parameters():
                    p.requires_grad = False
    return net


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
