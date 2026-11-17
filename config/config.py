"""ViLiS project configuration with automatic device detection."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _detect_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Project root: two levels up from this file (config/config.py -> vilis/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ViLiSConfig:
    """Central configuration for the ViLiS Hindi lip-reading system."""

    # ── Device ───────────────────────────────────────────────────────────
    DEVICE: str = field(default_factory=_detect_device)

    # ── Model hyper-parameters ───────────────────────────────────────────
    D_MODEL: int = 512
    NUM_LAYERS: int = 12
    NHEAD: int = 8
    CONV_KERNEL: int = 31
    DROPOUT: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────
    BATCH_SIZE: int = field(init=False)   # set in __post_init__
    LR: float = 3e-4
    WARMUP_STEPS: int = 500
    MAX_GRAD_NORM: float = 5.0
    EPOCHS: int = 100

    # ── Data ─────────────────────────────────────────────────────────────
    IMG_SIZE: int = 96
    MAX_FRAMES: int = 75
    VOCAB_SIZE: int = 4000

    # ── Paths ────────────────────────────────────────────────────────────
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    CHECKPOINT_DIR: Path = _PROJECT_ROOT / "checkpoints"
    ONNX_DIR: Path = _PROJECT_ROOT / "onnx_models"

    # ── Whisper model for pseudo-label generation ────────────────────────
    WHISPER_MODEL: str = field(init=False)  # set in __post_init__

    # ── Mixed precision ──────────────────────────────────────────────────
    USE_AMP: bool = field(init=False)  # set in __post_init__

    def __post_init__(self) -> None:
        is_cuda = self.DEVICE == "cuda"

        # Batch size: larger on CUDA where VRAM is plentiful
        self.BATCH_SIZE = 8 if is_cuda else 2

        # Whisper variant: large-v3 only practical on CUDA
        self.WHISPER_MODEL = "large-v3" if is_cuda else "medium"

        # AMP autocast works reliably only on CUDA
        self.USE_AMP = is_cuda

        # Ensure output directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.ONNX_DIR.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_config() -> ViLiSConfig:
    """Return a singleton ViLiSConfig instance."""
    return ViLiSConfig()
