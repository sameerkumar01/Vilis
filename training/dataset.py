"""PyTorch Dataset & DataLoader for ViLiS video speech recognition."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import get_config
from training.augmentations import VideoAugmentor
from utils.tokenizer import ViLiSTokenizer

logger = logging.getLogger(__name__)


class ViLiSDataset(Dataset):
    """Dataset loading preprocessed lip tensors (.npz or .npy) and transcripts from TSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: ViLiSTokenizer,
        max_frames: int = 75,
        is_training: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.is_training = is_training
        self.augmentor = VideoAugmentor(is_training=is_training)

        self.samples: List[Tuple[str, str, int]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            logger.warning("Manifest file not found: %s", self.manifest_path)
            return

        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    npz_path = row[0].strip()
                    transcript = row[1].strip()
                    n_frames = int(row[2]) if len(row) >= 3 else 0
                    self.samples.append((npz_path, transcript, n_frames))

        logger.info("Loaded %d samples from %s", len(self.samples), self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        npz_path_str, transcript, _ = self.samples[idx]
        npz_path = Path(npz_path_str)

        # Load lip frames
        if npz_path.exists():
            if npz_path.suffix == ".npz":
                data = np.load(npz_path)
                frames = data["lip_frames"]  # shape (T, 96, 96) or (T, H, W)
            else:
                frames = np.load(npz_path)
        else:
            # Fallback dummy frames if file missing during dry run
            frames = np.zeros((10, 96, 96), dtype=np.float32)

        # Ensure shape (T, 1, 96, 96) and float32 normalized [0, 1]
        if frames.ndim == 3:
            frames = frames[:, np.newaxis, :, :]  # (T, 1, 96, 96)

        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)

        # Truncate if longer than max_frames
        if len(frames) > self.max_frames:
            frames = frames[: self.max_frames]

        tensor_frames = torch.from_numpy(frames)

        # Augmentation
        tensor_frames = self.augmentor(tensor_frames)

        # Tokenize transcript
        token_ids = self.tokenizer.encode(transcript)
        target_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "frames": tensor_frames,
            "target": target_tensor,
            "input_length": tensor_frames.shape[0],
            "target_length": len(token_ids),
            "transcript": transcript,
        }


def pad_collate_fn(batch: List[dict]) -> dict:
    """Collate and pad variable-length video frames and token sequences."""
    batch_size = len(batch)

    input_lengths = torch.tensor([item["input_length"] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)

    max_t = int(input_lengths.max().item())
    max_l = int(target_lengths.max().item()) if target_lengths.max() > 0 else 1

    # Frames tensor: (B, max_t, 1, 96, 96)
    padded_frames = torch.zeros((batch_size, max_t, 1, 96, 96), dtype=torch.float32)
    # Targets tensor: (B, max_l)
    padded_targets = torch.zeros((batch_size, max_l), dtype=torch.long)

    transcripts = []
    for i, item in enumerate(batch):
        t_len = item["input_length"]
        padded_frames[i, :t_len] = item["frames"]

        l_len = item["target_length"]
        if l_len > 0:
            padded_targets[i, :l_len] = item["target"]

        transcripts.append(item["transcript"])

    return {
        "frames": padded_frames,
        "targets": padded_targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "transcripts": transcripts,
    }
