"""
Spatio-Temporal Visual Frontend for ViLiS.

Pipeline:
  (B, T, 1, 96, 96) → 3D-Conv → BN → ReLU → MaxPool
                     → per-frame ResNet-18 backbone → (B, T, 512)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SpatioTemporalFrontend(nn.Module):
    """3D conv + ResNet-18 visual feature extractor for lip-reading."""

    def __init__(self) -> None:
        super().__init__()

        # ---------- 3D convolutional stem ----------
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn3d = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        # ---------- 2D ResNet-18 backbone ----------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Patch conv1 to accept 64-channel feature maps instead of 3-ch RGB
        original_conv1 = resnet.conv1  # Conv2d(3, 64, 7, stride=2, padding=3)
        self.resnet_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        # Initialize by averaging pretrained weights across input channels
        with torch.no_grad():
            # original weight: (64, 3, 7, 7) → mean over dim-1 → (64, 1, 7, 7) → expand
            avg_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.resnet_conv1.weight.copy_(avg_weight.expand_as(self.resnet_conv1.weight))

        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1  # → 64
        self.resnet_layer2 = resnet.layer2  # → 128
        self.resnet_layer3 = resnet.layer3  # → 256
        self.resnet_layer4 = resnet.layer4  # → 512

        # avgpool & fc are intentionally removed; we use adaptive pooling ourselves
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1, H, W)  —  grayscale lip crops, H=W=96

        Returns:
            (B, T, 512)  —  per-frame visual features
        """
        B, T, C, H, W = x.shape

        # --- 3D conv expects (B, C, T, H, W) ---
        x = x.permute(0, 2, 1, 3, 4)           # (B, 1, T, H, W)
        x = self.conv3d(x)                      # (B, 64, T, H/2, W/2)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.pool3d(x)                      # (B, 64, T, H/4, W/4)

        # --- Process each frame through 2D ResNet ---
        # Reshape: merge batch & time
        _, C_out, T_out, H_out, W_out = x.shape
        x = x.permute(0, 2, 1, 3, 4)           # (B, T, 64, H', W')
        x = x.reshape(B * T_out, C_out, H_out, W_out)

        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)               # (B*T, 512, h, w)

        x = self.adaptive_pool(x)               # (B*T, 512, 1, 1)
        x = x.flatten(1)                        # (B*T, 512)
        x = x.view(B, T_out, -1)                # (B, T, 512)

        return x
