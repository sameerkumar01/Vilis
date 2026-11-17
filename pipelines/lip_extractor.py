"""Lip ROI extraction using MediaPipe FaceMesh.

Extracts and aligns a 96×96 grayscale lip crop from each video frame.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from config.config import get_config

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh landmark indices that outline the lips.
_LIP_LANDMARKS: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
]

# Mouth corners used for affine roll correction.
_LEFT_CORNER = 61
_RIGHT_CORNER = 291

# Padding factor around the lip bounding box before cropping.
_PAD_FACTOR = 0.3


class LipExtractor:
    """Extract 96×96 grayscale lip crops from video frames via MediaPipe."""

    def __init__(
        self,
        img_size: Optional[int] = None,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        cfg = get_config()
        self.img_size = img_size or cfg.IMG_SIZE

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ── public API ────────────────────────────────────────────────────────

    def extract_lips_from_video(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Return a list of 96×96 grayscale lip frames from *video_path*.

        Parameters
        ----------
        video_path : str
            Path to the source video.
        start_sec / end_sec : float, optional
            Time window (seconds) to extract from.  ``None`` means
            beginning / end of the video.

        Returns
        -------
        list[np.ndarray]
            Each element has shape ``(img_size, img_size)`` with dtype
            ``uint8``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(start_sec * fps)
        end_frame = (
            int(end_sec * fps)
            if end_sec is not None
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: List[np.ndarray] = []
        last_good: Optional[np.ndarray] = None
        idx = start_frame

        while idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            lip = self.extract_lips_from_frame(frame)
            if lip is not None:
                frames.append(lip)
                last_good = lip
            elif last_good is not None:
                # Interpolate: repeat the last good frame.
                frames.append(last_good.copy())
            # else: skip — no face detected and no prior frame.

            idx += 1

        cap.release()
        return frames

    def extract_lips_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single 96×96 grayscale lip crop from *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        np.ndarray or None
            ``(img_size, img_size)`` uint8 grayscale, or ``None`` if no
            face was detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Gather lip landmark pixel coordinates.
        pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in _LIP_LANDMARKS],
            dtype=np.float32,
        )

        # Mouth corners for roll correction.
        left = np.array(
            [landmarks[_LEFT_CORNER].x * w, landmarks[_LEFT_CORNER].y * h],
            dtype=np.float32,
        )
        right = np.array(
            [landmarks[_RIGHT_CORNER].x * w, landmarks[_RIGHT_CORNER].y * h],
            dtype=np.float32,
        )

        # ── Affine roll correction ───────────────────────────────────────
        angle = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
        center = ((left + right) / 2).astype(np.float32)
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)

        # Rotate the whole frame.
        rotated = cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

        # Transform lip points to rotated space.
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])  # (N, 3)
        pts_rot = (rot_mat @ pts_h.T).T  # (N, 2)

        # ── Bounding box with padding ────────────────────────────────────
        x_min, y_min = pts_rot.min(axis=0)
        x_max, y_max = pts_rot.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        pad_x, pad_y = bw * _PAD_FACTOR, bh * _PAD_FACTOR

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        crop = rotated[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to grayscale and resize.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return resized

    # ── context-manager / cleanup ─────────────────────────────────────────

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "LipExtractor":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
