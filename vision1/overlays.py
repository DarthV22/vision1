"""OpenCV annotations only; all measurements come from the processing pipeline."""

import cv2
import numpy as np

from vision1.config import Config
from vision1.geometry import Point, speed_direction
from vision1.hands import HAND_CONNECTIONS, Hand
from vision1.motion import FlowResult
from vision1.tracking import Track

ACCENT = (200, 220, 55)  # BGR cyan
TARGET = (85, 205, 250)
MUTED = (125, 145, 155)


def pixel(point: Point) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def text(image: np.ndarray, value: str, origin: tuple[int, int], color=ACCENT) -> None:
    x, y = origin
    x = max(4, min(x, image.shape[1] - 12))
    y = max(16, min(y, image.shape[0] - 5))
    (width, height), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    x = max(4, min(x, image.shape[1] - width - 5))
    cv2.rectangle(image, (x - 3, y - height - 5), (x + width + 3, y + 4), (20, 27, 31), -1)
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def render(
    image: np.ndarray, tracks: list[Track], hands: list[Hand], flow: FlowResult,
    selected: int | None, config: Config, now: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    if config.grid:
        for x in range(width // 8, width, max(1, width // 8)):
            cv2.line(image, (x, 0), (x, height), (45, 52, 53), 1)
        for y in range(height // 6, height, max(1, height // 6)):
            cv2.line(image, (0, y), (width, y), (45, 52, 53), 1)
        for x in range(width // 4, width, max(1, width // 4)):
            text(image, f"{x}px", (x + 3, height - 10), MUTED)
        for y in range(height // 3, height, max(1, height // 3)):
            text(image, f"{y}px", (8, y - 4), MUTED)
        cv2.drawMarker(image, (width // 2, height // 2), MUTED, cv2.MARKER_CROSS, 18, 1)
    if config.vectors:
        for start, end in flow.vectors:
            cv2.arrowedLine(image, pixel(start), pixel(end), (110, 120, 70), 1, cv2.LINE_AA, tipLength=0.3)
    for track in tracks:
        box = track.box
        x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
        color = TARGET if track.id == selected else ACCENT
        stale = now - track.last_seen > 0.25
        if stale:
            color = MUTED
        if config.trajectories and len(track.history) > 1:
            points = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
        if config.boxes or track.id == selected:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if track.id == selected else 1)
            for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(image, (x, y), (x + dx * 12, y), color, 2)
                cv2.line(image, (x, y), (x, y + dy * 12), color, 2)
            label = f"#{track.id:02d} {track.detection.label} {track.detection.confidence:.0%}"
            if stale:
                label += " | last seen"
            text(image, label, (x1, y1 - 10), color)
        center = box.center
        if config.vectors and track.observations > 1:
            # A 0.15-second velocity vector, capped to keep it on the viewport.
            vector = np.array(track.vector) * 0.15
            length = float(np.linalg.norm(vector))
            if length > 100:
                vector *= 100 / length
            end = center[0] + vector[0], center[1] + vector[1]
            cv2.arrowedLine(image, pixel(center), pixel(end), color, 2, cv2.LINE_AA, tipLength=0.25)
        if track.id == selected:
            cv2.drawMarker(image, pixel(center), TARGET, cv2.MARKER_CROSS, 28, 1)
            speed, _ = speed_direction(track.vector)
            motion = f"{speed:.1f} px/s" if track.observations > 1 else "velocity --"
            text(image, f"{center[0]:.0f}, {center[1]:.0f} px | {motion}", (x1, y2 + 20), TARGET)
    for hand in hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pixel(hand.points[a]), pixel(hand.points[b]), ACCENT, 2, cv2.LINE_AA)
        for index, point in enumerate(hand.points):
            cv2.circle(image, pixel(point), 4 if index in (4, 8, 12, 16, 20) else 2, TARGET, -1, cv2.LINE_AA)
        text(image, f"{hand.side} | {hand.gesture}", pixel(hand.points[0]), TARGET)
    return image
