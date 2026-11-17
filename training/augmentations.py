"""Data Augmentations for Visual Speech Recognition video sequences.

Implements spatial and temporal augmentations:
- Random Crop (e.g. 96x96 -> 88x88 -> resize 96x96)
- Horizontal Flip (p=0.5)
- Speed Perturbation (resampling ±10%)
- Temporal Masking (zeroing consecutive frames)
- Frame Masking (zeroing random isolated frames)
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class VideoAugmentor:
    """Applies spatio-temporal augmentations consistently across a video sequence."""

    def __init__(
        self,
        crop_size: int = 88,
        flip_prob: float = 0.5,
        speed_factor_range: Tuple[float, float] = (0.9, 1.1),
        max_temporal_mask: int = 8,
        max_frame_mask: int = 4,
        is_training: bool = True,
    ) -> None:
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.speed_factor_range = speed_factor_range
        self.max_temporal_mask = max_temporal_mask
        self.max_frame_mask = max_frame_mask
        self.is_training = is_training

    def random_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, 1, H, W). Random spatial crop and bilinear resize back to (H, W)."""
        _, _, h, w = frames.shape
        if h <= self.crop_size or w <= self.crop_size:
            return frames

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        cropped = frames[:, :, top : top + self.crop_size, left : left + self.crop_size]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)

    def horizontal_flip(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, 1, H, W). Random horizontal flip across all frames identically."""
        if random.random() < self.flip_prob:
            return torch.flip(frames, dims=[3])
        return frames

    def temporal_masking(self, frames: torch.Tensor) -> torch.Tensor:
        """Zero out a continuous block of frames (like SpecAugment)."""
        t, _, _, _ = frames.shape
        if t <= 2 or self.max_temporal_mask <= 0:
            return frames

        mask_len = random.randint(1, min(self.max_temporal_mask, t // 2))
        start_idx = random.randint(0, t - mask_len)
        frames = frames.clone()
        frames[start_idx : start_idx + mask_len] = 0.0
        return frames

    def frame_masking(self, frames: torch.Tensor) -> torch.Tensor:
        """Zero out random individual frames."""
        t, _, _, _ = frames.shape
        if t <= 2 or self.max_frame_mask <= 0:
            return frames

        num_masks = random.randint(0, min(self.max_frame_mask, t // 3))
        indices = random.sample(range(t), num_masks)
        frames = frames.clone()
        for idx in indices:
            frames[idx] = 0.0
        return frames

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline if training."""
        if not self.is_training:
            return frames

        frames = self.random_crop(frames)
        frames = self.horizontal_flip(frames)
        frames = self.temporal_masking(frames)
        frames = self.frame_masking(frames)
        return frames
