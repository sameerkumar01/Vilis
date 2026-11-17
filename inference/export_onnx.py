"""Export trained ViLiS model to ONNX for fast deployment on Mac, Windows, Linux, or mobile."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config.config import get_config
from model.vilis_model import ViLiSModel
from utils.tokenizer import ViLiSTokenizer

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str | Path | None,
    output_path: str | Path,
    sp_model_path: str | Path | None = None,
    opset_version: int = 17,
) -> Path:
    cfg = get_config()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = ViLiSTokenizer(sp_model_path)
    cfg.VOCAB_SIZE = tokenizer.vocab_size

    model = ViLiSModel(cfg)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)

    # Dummy input: (Batch=1, Time=50, Channels=1, H=96, W=96)
    dummy_input = torch.randn(1, 50, 1, 96, 96, dtype=torch.float32)

    logger.info("Exporting to ONNX at %s (opset %d)...", output_path, opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["video_frames"],
        output_names=["logits"],
        dynamic_axes={
            "video_frames": {0: "batch_size", 1: "time_steps"},
            "logits": {0: "batch_size", 1: "time_steps"},
        },
    )
    logger.info("ONNX export successful: %s", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ViLiS to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="onnx_models/vilis_hindi.onnx")
    parser.add_argument("--sp-model", type=str, default=None)
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.sp_model)
