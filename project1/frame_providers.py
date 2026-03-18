from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np


class FrameProvider(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def is_open(self) -> bool: ...

    def release(self) -> None: ...


class StaticImageSource:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        img: np.ndarray | None = cv.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not decode image at {path}")

        self._image: np.ndarray = img
        self._is_open: bool = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._is_open:
            return True, self._image.copy()

        return False, None

    def is_open(self) -> bool:
        return self._is_open

    def release(self) -> None:
        self._is_open = False


class VideoSource:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self._cap: cv.VideoCapture = cv.VideoCapture(str(path))

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        success: bool
        frame: np.ndarray | None
        success, frame = self._cap.read()

        return success, frame

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()


class CameraSource:
    def __init__(self, device_id: int) -> None:
        self._cap: cv.VideoCapture = cv.VideoCapture(device_id)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {device_id}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        success: bool
        frame: np.ndarray | None
        success, frame = self._cap.read()

        return success, frame

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
