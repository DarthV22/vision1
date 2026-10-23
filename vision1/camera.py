"""Single capture thread with a one-frame mailbox, never a video queue."""

from dataclasses import dataclass
import logging
import sys
from threading import Condition, Event, Thread
from time import monotonic

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    timestamp: float
    sequence: int


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.index, self.width, self.height = index, width, height
        self._condition = Condition()
        self._stop = Event()
        self._latest: CameraFrame | None = None
        self._error: str | None = None
        self._thread = Thread(target=self._capture, name="camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self, after: int, timeout: float = 0.1) -> CameraFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._stop.is_set() or self._error is not None
                or (self._latest is not None and self._latest.sequence > after),
                timeout,
            )
            if self._error:
                raise OSError(self._error)
            packet = self._latest
            return packet if packet is not None and packet.sequence > after else None

    def stop(self) -> bool:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        return not self._thread.is_alive()

    def _capture(self) -> None:
        capture = None
        try:
            # On Windows, MSMF can open HP/IR cameras without delivering frames.
            # Prefer DirectShow there; retain the native default on other systems.
            if sys.platform == "win32":
                capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(self.index)
            else:
                capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise OSError(f"Camera {self.index} is unavailable. Check its index, permissions and other apps.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Advisory: some backends ignore this.
            sequence, misses = 0, 0
            while not self._stop.is_set():
                ok, image = capture.read()
                if not ok or image is None:
                    misses += 1
                    if misses >= 10:
                        raise OSError("Camera disconnected or stopped delivering frames.")
                    self._stop.wait(0.03)
                    continue
                misses = 0
                sequence += 1
                with self._condition:
                    self._latest = CameraFrame(image, monotonic(), sequence)
                    self._condition.notify()
        except Exception as exc:
            log.exception("Camera acquisition failed")
            with self._condition:
                self._error = str(exc)
                self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()
