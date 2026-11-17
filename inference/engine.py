"""ViLiS Inference Engine for video and real-time lip reading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from config.config import get_config
from model.vilis_model import ViLiSModel
from pipelines.lip_extractor import LipExtractor
from utils.tokenizer import ViLiSTokenizer

logger = logging.getLogger(__name__)


class ViLiSInferenceEngine:
    """End-to-end inference engine: Raw Video -> Lip Extraction -> Conformer -> Hindi Text."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        sp_model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = get_config()
        self.device = torch.device(device or self.cfg.DEVICE)
        self.tokenizer = ViLiSTokenizer(sp_model_path)
        self.cfg.VOCAB_SIZE = self.tokenizer.vocab_size

        self.model = ViLiSModel(self.cfg).to(self.device)
        self.model.eval()

        self.extractor = LipExtractor()

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load trained model weights."""
        logger.info("Loading checkpoint from %s", path)
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            self.model.load_state_dict(ckpt)
        logger.info("Model weights loaded successfully.")

    @torch.no_grad()
    def predict_frames(self, lip_frames: np.ndarray | torch.Tensor) -> Tuple[str, List[int]]:
        """Predict Hindi text from extracted lip frames.

        lip_frames: shape (T, 96, 96) or (T, 1, 96, 96)
        """
        if isinstance(lip_frames, np.ndarray):
            if lip_frames.ndim == 3:
                lip_frames = lip_frames[:, np.newaxis, :, :]
            if lip_frames.dtype == np.uint8:
                lip_frames = lip_frames.astype(np.float32) / 255.0
            tensor_frames = torch.from_numpy(lip_frames).float()
        else:
            tensor_frames = lip_frames.float()

        if tensor_frames.ndim == 4:
            # Add batch dimension: (1, T, 1, 96, 96)
            tensor_frames = tensor_frames.unsqueeze(0)

        tensor_frames = tensor_frames.to(self.device)

        # Forward pass
        logits = self.model(tensor_frames)  # (1, T, vocab_size)
        predictions = torch.argmax(logits, dim=-1).squeeze(0)  # (T,)
        token_ids = predictions.cpu().tolist()

        # CTC greedy decode
        text = self.tokenizer.ctc_greedy_decode(token_ids)
        if not text:
            text = "[no prediction — needs more training data]"

        return text, token_ids

    def predict_video(self, video_path: str | Path) -> Tuple[str, List[int], int]:
        """Process video file, extract lips, and return Hindi prediction."""
        logger.info("Processing video: %s", video_path)
        lip_frames = self.extractor.extract_lips_from_video(video_path)
        if len(lip_frames) == 0:
            return "[no face/lips detected in video]", [], 0

        frames_array = np.array(lip_frames, dtype=np.float32)
        text, token_ids = self.predict_frames(frames_array)
        return text, token_ids, len(lip_frames)
