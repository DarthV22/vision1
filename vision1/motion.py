"""Sparse pyramidal Lucas-Kanade flow with forward/backward rejection."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from vision1.geometry import Point


@dataclass(frozen=True)
class FlowResult:
    vectors: list[tuple[Point, Point]] = field(default_factory=list)
    velocity: Point = (0.0, 0.0)
    valid: bool = False


class OpticalFlow:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.timestamp = 0.0

    def reset(self) -> None:
        self.previous, self.points = None, None

    def update(self, frame: np.ndarray, now: float) -> FlowResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = FlowResult()
        dt = now - self.timestamp
        if (
            self.previous is not None and self.previous.shape == gray.shape
            and self.points is not None and 0 < dt < 0.5
        ):
            current, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous, gray, self.points, None, winSize=(21, 21), maxLevel=3,
            )
            if current is not None and status is not None:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, current, None)
                if backward is not None and back_status is not None:
                    good = (
                        (status.ravel() == 1) & (back_status.ravel() == 1)
                        & (np.linalg.norm(self.points - backward, axis=2).ravel() < 1.5)
                        & np.isfinite(current).all(axis=(1, 2))
                    )
                    old = self.points[good].reshape(-1, 2)
                    new = current[good].reshape(-1, 2)
                    if len(new):
                        vector = np.median((new - old) / dt, axis=0)
                        vectors = [(tuple(map(float, a)), tuple(map(float, b))) for a, b in zip(old, new)]
                        result = FlowResult(vectors, tuple(map(float, vector)), True)
                    self.points = new.reshape(-1, 1, 2)
        if self.points is None or len(self.points) < 40 or not result.valid:
            self.points = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.02, minDistance=12, blockSize=7,
            )
        self.previous, self.timestamp = gray, now
        return result
