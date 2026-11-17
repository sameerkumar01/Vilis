"""Whisper-based transcription pipeline for Hindi video data.

Generates word-level timestamped transcripts and writes them to a CSV
that can be consumed by :class:`pipelines.dataset_builder.DatasetBuilder`.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import whisper

from config.config import get_config

logger = logging.getLogger(__name__)

_VIDEO_EXTS: Set[str] = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
_CSV_COLUMNS: List[str] = ["video_path", "start_time", "end_time", "text"]

# Minimum segment energy (RMS) to keep — acts as a simple VAD gate.
_MIN_RMS_ENERGY = 0.005


class WhisperTranscriber:
    """Transcribe Hindi speech from video files using OpenAI Whisper."""

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None) -> None:
        cfg = get_config()
        self._model_size = model_size or cfg.WHISPER_MODEL
        self._device = device or cfg.DEVICE
        self._model: Optional[whisper.Whisper] = None

    # ── lazy model loading ────────────────────────────────────────────────

    def _load_model(self) -> whisper.Whisper:
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' on %s …", self._model_size, self._device,
            )
            self._model = whisper.load_model(self._model_size, device=self._device)
        return self._model

    # ── public API ────────────────────────────────────────────────────────

    def transcribe_video(self, video_path: str) -> Dict:
        """Transcribe *video_path* and return a result dict.

        The result follows the Whisper output format with word-level
        timestamps (``word_timestamps=True``).

        Parameters
        ----------
        video_path : str
            Path to a video (audio is extracted internally by Whisper).

        Returns
        -------
        dict
            Whisper result dict with keys ``text``, ``segments``, and
            ``language``.  Each segment contains a ``words`` list when
            available.
        """
        model = self._load_model()
        result = model.transcribe(
            video_path,
            language="hi",
            task="transcribe",
            word_timestamps=True,
            verbose=False,
        )
        return result

    def transcribe_batch(
        self,
        video_dir: str,
        output_csv: str,
    ) -> Path:
        """Transcribe every video in *video_dir* and write results to CSV.

        The CSV has columns ``video_path, start_time, end_time, text``.
        If *output_csv* already exists, previously processed videos are
        skipped so the job can be safely stopped and resumed.

        Parameters
        ----------
        video_dir : str
            Directory containing video files.
        output_csv : str
            Destination CSV path.

        Returns
        -------
        Path
            Resolved path to the written CSV.
        """
        video_dir_path = Path(video_dir)
        output_path = Path(output_csv)

        # Discover video files.
        videos = sorted(
            p for p in video_dir_path.rglob("*") if p.suffix.lower() in _VIDEO_EXTS
        )
        if not videos:
            logger.warning("No video files found in %s", video_dir)
            return output_path

        # Load already-processed paths for resume support.
        done: Set[str] = set()
        if output_path.exists():
            with open(output_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    done.add(row["video_path"])

        file_exists = output_path.exists()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()

            for idx, vpath in enumerate(videos, 1):
                vpath_str = str(vpath)
                if vpath_str in done:
                    logger.debug("Skipping already-processed %s", vpath_str)
                    continue

                logger.info("[%d/%d] Transcribing %s …", idx, len(videos), vpath.name)
                try:
                    result = self.transcribe_video(vpath_str)
                except Exception:
                    logger.exception("Failed to transcribe %s — skipping.", vpath.name)
                    continue

                rows = _result_to_rows(vpath_str, result)
                for row in rows:
                    writer.writerow(row)
                fh.flush()

        logger.info("Transcription complete → %s", output_path)
        return output_path


# ── helpers ───────────────────────────────────────────────────────────────


def _is_silent(audio: np.ndarray, threshold: float = _MIN_RMS_ENERGY) -> bool:
    """Return True if the audio segment is near-silent."""
    if audio is None or len(audio) == 0:
        return True
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < threshold


def _result_to_rows(video_path: str, result: Dict) -> List[Dict[str, str]]:
    """Convert Whisper result segments into CSV-ready row dicts.

    Segments whose text is empty or only whitespace are discarded
    (simple VAD-style filter).
    """
    rows: List[Dict[str, str]] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        # If word-level timestamps exist, use first-word start / last-word end.
        words = seg.get("words")
        if words:
            start = words[0].get("start", seg["start"])
            end = words[-1].get("end", seg["end"])
        else:
            start = seg["start"]
            end = seg["end"]

        rows.append(
            {
                "video_path": video_path,
                "start_time": f"{start:.3f}",
                "end_time": f"{end:.3f}",
                "text": text,
            }
        )
    return rows
