"""Dataset builder: converts transcripts + videos into training-ready data.

Reads ``transcripts.csv`` (produced by :class:`WhisperTranscriber`),
extracts lip-ROI frames for each segment, saves them as ``.npz`` files,
and produces train/val TSV manifests.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config.config import get_config
from pipelines.lip_extractor import LipExtractor

logger = logging.getLogger(__name__)

_MANIFEST_COLUMNS: List[str] = ["npz_path", "transcript", "num_frames"]

# Minimum / maximum frame counts for a segment to be kept.
_MIN_FRAMES = 5


class DatasetBuilder:
    """Build ``.npz`` frame caches and TSV manifests from transcripts."""

    def __init__(
        self,
        val_ratio: float = 0.10,
        seed: int = 42,
        max_frames: Optional[int] = None,
    ) -> None:
        cfg = get_config()
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_frames = max_frames or cfg.MAX_FRAMES
        self._extractor = LipExtractor()

    # ── public API ────────────────────────────────────────────────────────

    def build(self, data_dir: str, output_dir: str) -> Tuple[Path, Path]:
        """Run the full pipeline.

        Parameters
        ----------
        data_dir : str
            Directory containing ``transcripts.csv`` and the source videos.
        output_dir : str
            Root directory for outputs.  Sub-directories ``cache/`` and
            ``manifests/`` are created underneath.

        Returns
        -------
        tuple[Path, Path]
            Paths to ``train.tsv`` and ``val.tsv``.
        """
        data_path = Path(data_dir)
        out_path = Path(output_dir)

        cache_dir = out_path / "cache"
        manifest_dir = out_path / "manifests"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        transcript_csv = data_path / "transcripts.csv"
        if not transcript_csv.exists():
            raise FileNotFoundError(
                f"transcripts.csv not found at {transcript_csv}"
            )

        entries = self._read_transcripts(transcript_csv)
        logger.info("Read %d transcript entries.", len(entries))

        # ── Extract and cache lip frames ──────────────────────────────────
        records: List[dict] = []
        with self._extractor:
            for idx, entry in enumerate(entries, 1):
                video_path = entry["video_path"]
                start = float(entry["start_time"])
                end = float(entry["end_time"])
                text = entry["text"]

                # Deterministic npz filename.
                key = f"{video_path}_{start:.3f}_{end:.3f}"
                digest = hashlib.md5(key.encode()).hexdigest()[:12]
                npz_name = f"{digest}.npz"
                npz_path = cache_dir / npz_name

                if npz_path.exists():
                    data = np.load(npz_path)
                    n_frames = data["frames"].shape[0]
                else:
                    try:
                        frames = self._extractor.extract_lips_from_video(
                            video_path, start_sec=start, end_sec=end,
                        )
                    except Exception:
                        logger.exception(
                            "[%d/%d] Failed to extract lips from %s (%.2f–%.2f) — skipping.",
                            idx, len(entries), video_path, start, end,
                        )
                        continue

                    n_frames = len(frames)
                    if n_frames == 0:
                        logger.debug("No lip frames for %s — skipping.", key)
                        continue

                    np.savez_compressed(npz_path, frames=np.stack(frames))

                # ── Filter by frame count ────────────────────────────────
                if n_frames < _MIN_FRAMES:
                    logger.debug("Too few frames (%d) for %s — skipping.", n_frames, key)
                    continue
                if n_frames > self.max_frames:
                    logger.debug("Too many frames (%d) for %s — skipping.", n_frames, key)
                    continue

                records.append(
                    {
                        "npz_path": str(npz_path),
                        "transcript": text,
                        "num_frames": str(n_frames),
                    }
                )

                if idx % 100 == 0 or idx == len(entries):
                    logger.info("[%d/%d] segments processed.", idx, len(entries))

        logger.info("Kept %d segments after filtering.", len(records))

        # ── Train / val split ─────────────────────────────────────────────
        random.seed(self.seed)
        random.shuffle(records)
        split_idx = max(1, int(len(records) * (1 - self.val_ratio)))
        train_records = records[:split_idx]
        val_records = records[split_idx:]

        train_tsv = manifest_dir / "train.tsv"
        val_tsv = manifest_dir / "val.tsv"
        self._write_manifest(train_tsv, train_records)
        self._write_manifest(val_tsv, val_records)

        logger.info(
            "Manifests written — train: %d, val: %d → %s",
            len(train_records), len(val_records), manifest_dir,
        )
        return train_tsv, val_tsv

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_transcripts(csv_path: Path) -> List[dict]:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader)

    @staticmethod
    def _write_manifest(path: Path, records: List[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=_MANIFEST_COLUMNS, delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(records)
