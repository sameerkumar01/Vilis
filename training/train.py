"""Training script for ViLiS Hindi Visual Speech Recognition.

Supports Apple Silicon MPS, NVIDIA CUDA (with AMP), and CPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import get_config
from model.vilis_model import ViLiSModel
from training.dataset import ViLiSDataset, pad_collate_fn
from training.scheduler import WarmupCosineScheduler
from utils.metrics import compute_cer, compute_wer
from utils.tokenizer import ViLiSTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    max_grad_norm: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        frames = batch["frames"].to(device)  # (B, T, 1, 96, 96)
        targets = batch["targets"].to(device)  # (B, L)
        input_lengths = batch["input_lengths"].to(device)  # (B,)
        target_lengths = batch["target_lengths"].to(device)  # (B,)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(frames)  # (B, T, vocab_size)
                # CTC Loss expects log_probs as (T, B, C)
                log_probs = logits.transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            log_probs = logits.transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 10 == 0:
            logger.info("Batch %d/%d | Loss: %.4f | LR: %.6f", step + 1, len(loader), loss.item(), scheduler.get_last_lr()[0])

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: ViLiSTokenizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        references = batch["transcripts"]

        logits = model(frames)
        log_probs = logits.transpose(0, 1)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()

        # Greedy CTC decoding
        predictions = torch.argmax(logits, dim=-1)  # (B, T)
        for i in range(len(references)):
            pred_ids = predictions[i, : input_lengths[i]].cpu().tolist()
            pred_text = tokenizer.ctc_greedy_decode(pred_ids)
            ref_text = references[i]

            cer = compute_cer(ref_text, pred_text)
            wer = compute_wer(ref_text, pred_text)

            total_cer += cer
            total_wer += wer
            total_samples += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_cer = total_cer / max(1, total_samples)
    avg_wer = total_wer / max(1, total_samples)
    return avg_loss, avg_cer, avg_wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViLiS Hindi Lip Reading Model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sp-model", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.DEVICE)
    logger.info("Using device: %s", device)

    # Tokenizer
    tokenizer = ViLiSTokenizer(args.sp_model)
    logger.info("Tokenizer loaded. Vocab size: %d", tokenizer.vocab_size)

    # Update vocab size in config
    cfg.VOCAB_SIZE = tokenizer.vocab_size

    # Model
    model = ViLiSModel(cfg).to(device)
    logger.info("ViLiS model initialized with %d parameters", model.count_parameters())

    # Datasets
    train_manifest = cfg.DATA_DIR / "processed" / "manifests" / "train.tsv"
    val_manifest = cfg.DATA_DIR / "processed" / "manifests" / "val.tsv"

    batch_size = args.batch_size or cfg.BATCH_SIZE
    train_dataset = ViLiSDataset(train_manifest, tokenizer, max_frames=cfg.MAX_FRAMES, is_training=True)
    val_dataset = ViLiSDataset(val_manifest, tokenizer, max_frames=cfg.MAX_FRAMES, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=0 if cfg.DEVICE == "mps" else 2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.WARMUP_STEPS, total_steps=total_steps)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    best_cer = float("inf")
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, use_amp, scaler, cfg.MAX_GRAD_NORM
        )
        val_loss, val_cer, val_wer = evaluate(model, val_loader, tokenizer, criterion, device)

        logger.info(
            "Epoch %d Results | Train Loss: %.4f | Val Loss: %.4f | Val CER: %.2f%% | Val WER: %.2f%%",
            epoch + 1, train_loss, val_loss, val_cer * 100, val_wer * 100
        )

        # Save checkpoint
        checkpoint_path = cfg.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer": val_cer,
                "val_wer": val_wer,
            },
            checkpoint_path,
        )

        if val_cer < best_cer:
            best_cer = val_cer
            best_path = cfg.CHECKPOINT_DIR / "best.pt"
            torch.save(model.state_dict(), best_path)
            logger.info("Saved new best model to %s (CER: %.2f%%)", best_path, best_cer * 100)


if __name__ == "__main__":
    main()
