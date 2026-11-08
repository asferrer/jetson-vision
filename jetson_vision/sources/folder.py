"""Folder of images (or one image file) as a finite, non-live source. Files are read in sorted name order."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from jetson_vision.interfaces import SourceError
from jetson_vision.sources.base import Pacer, SourceBase, check_mode, stamp_ns
from jetson_vision.types import Frame

log = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FolderSource(SourceBase):
    """mode 'max' delivers as fast as the consumer takes them; 'paced' releases one frame every 1/fps seconds.

    source_stamp is index / fps in nanoseconds in both modes. Unreadable images are skipped with a warning.
    """

    live = False

    def __init__(self, path: str | Path, *, mode: str = "max", fps: float = 30.0,
                 source_id: str | None = None) -> None:
        p = Path(path)
        super().__init__(source_id or f"folder:{p.name}")
        self.path, self.fps, self.mode = p, fps, check_mode(mode)
        self._files: list[Path] = []
        self._index = 0
        self._pacer = Pacer(fps if mode == "paced" else None)

    def open(self) -> None:
        if self.path.is_dir():
            files = sorted(f for f in self.path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif self.path.is_file():
            files = [self.path]
        else:
            raise SourceError(f"folder source: '{self.path}' does not exist")
        if not files:
            raise SourceError(f"folder source: no images in '{self.path}'")
        self._files = files

    def read(self) -> Frame | None:
        while self._index < len(self._files):
            index, file = self._index, self._files[self._index]
            self._index += 1
            self._pacer.wait(index)
            img = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("skipping unreadable image %s", file.name)
                continue
            return self.make_frame(img, stamp_ns(index, self.fps))
        return None

    def close(self) -> None:
        self._files = []
