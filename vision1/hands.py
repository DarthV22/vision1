"""MediaPipe Tasks hand landmarks and conservative geometric gestures."""

from dataclasses import dataclass
from math import acos, degrees, hypot
from pathlib import Path

import numpy as np

from vision1.geometry import Point

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def joint_angle(a: Point, b: Point, c: Point) -> float:
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    length = hypot(*u) * hypot(*v)
    if length < 1e-8:
        return 0.0
    return degrees(acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / length))))


def recognise_gesture(points: list[Point]) -> str:
    """Use image-plane joint angles and palm-relative distances, not screen-up tests."""
    if len(points) != 21:
        raise ValueError("A hand needs 21 landmarks")
    palm = distance(points[0], points[9])
    if palm < 1e-6:
        return "Unknown"
    if distance(points[4], points[8]) / palm < 0.32:
        return "Pinch"
    extended = [
        joint_angle(points[base], points[base + 1], points[base + 3]) > 155
        and distance(points[base + 3], points[0]) > distance(points[base + 1], points[0]) * 1.12
        for base in (5, 9, 13, 17)
    ]
    thumb = (
        joint_angle(points[2], points[3], points[4]) > 145
        and distance(points[4], points[17]) / palm > 0.8
    )
    if all(extended) and thumb:
        return "Open palm"
    if extended[0] and not any(extended[1:]):
        return "Pointing"
    if not any(extended):
        return "Closed fist"
    return "Unknown"


@dataclass(frozen=True)
class Hand:
    points: list[Point]
    side: str
    gesture: str


class HandProcessor:
    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.last_timestamp = -1

    def detect(self, rgb: np.ndarray, timestamp: float) -> list[Hand]:
        # Tasks VIDEO requires strictly increasing millisecond timestamps.
        timestamp_ms = max(self.last_timestamp + 1, int(timestamp * 1000))
        self.last_timestamp = timestamp_ms
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        height, width = rgb.shape[:2]
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = [(p.x * width, p.y * height) for p in landmarks]
            side = handedness[0].category_name if handedness else "Hand"
            hands.append(Hand(points, side, recognise_gesture(points)))
        return hands

    def close(self) -> None:
        self.landmarker.close()


class GestureControl:
    """Dwell and cooldown prevent repeated actions while a gesture is held."""

    def __init__(self) -> None:
        self.candidate = "Unknown"
        self.since = 0.0
        self.fired = False
        self.last_action = float("-inf")

    def update(self, hands: list[Hand], now: float) -> tuple[str, Point | None] | None:
        # One hand avoids ambiguity between simultaneous conflicting gestures.
        hand = hands[0] if len(hands) == 1 else None
        gesture = hand.gesture if hand else "Unknown"
        if gesture != self.candidate:
            self.candidate, self.since, self.fired = gesture, now, False
        if hand is None or self.fired or now - self.since < 0.35 or now - self.last_action < 1.0:
            return None
        if gesture not in ("Pinch", "Open palm"):
            return None
        self.fired, self.last_action = True, now
        if gesture == "Open palm":
            return "clear", None
        a, b = hand.points[4], hand.points[8]
        return "select", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
