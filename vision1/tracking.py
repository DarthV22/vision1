"""Class-aware, greedy IoU association with time-based expiry.

IDs persist across short misses, but are not re-identification identities.
Only detector observations update measured velocity and trajectories.
"""

from collections import deque
from dataclasses import dataclass, field
from math import hypot

from vision1.geometry import Box, Point, iou, velocity


@dataclass(frozen=True)
class Detection:
    box: Box
    class_id: int
    label: str
    confidence: float


@dataclass
class Track:
    id: int
    detection: Detection
    last_seen: float
    vector: Point = (0.0, 0.0)
    displacement: Point = (0.0, 0.0)
    history: deque[Point] = field(default_factory=lambda: deque(maxlen=60))
    observations: int = 1

    @property
    def box(self) -> Box:
        return self.detection.box

    def predicted_box(self, now: float) -> Box:
        dt = min(max(now - self.last_seen, 0.0), 0.5)
        return self.box.translated(self.vector[0] * dt, self.vector[1] * dt)


class Tracker:
    def __init__(self, max_age: float = 0.8, min_iou: float = 0.15) -> None:
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def expire(self, now: float) -> list[Track]:
        self.tracks = {key: t for key, t in self.tracks.items() if now - t.last_seen <= self.max_age}
        return list(self.tracks.values())

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        self.expire(now)
        candidates = []
        for key, track in self.tracks.items():
            predicted = track.predicted_box(now)
            for index, detection in enumerate(detections):
                if track.detection.class_id != detection.class_id:
                    continue
                overlap = iou(predicted, detection.box)
                # A small displacement gate covers fast movement of small objects.
                distance = hypot(
                    predicted.center[0] - detection.box.center[0],
                    predicted.center[1] - detection.box.center[1],
                )
                gate = max(20.0, min(predicted.width, predicted.height) * 0.6)
                if overlap >= self.min_iou or distance <= gate:
                    candidates.append((overlap - 0.1 * distance / gate, key, index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, key, index in sorted(candidates, reverse=True):
            if key in used_tracks or index in used_detections:
                continue
            used_tracks.add(key)
            used_detections.add(index)
            track = self.tracks[key]
            detection = detections[index]
            old_center, center = track.box.center, detection.box.center
            track.vector = velocity(old_center, center, now - track.last_seen)
            track.displacement = (center[0] - old_center[0], center[1] - old_center[1])
            track.detection = detection
            track.last_seen = now
            track.history.append(center)
            track.observations += 1
        for index, detection in enumerate(detections):
            if index not in used_detections:
                track = Track(self.next_id, detection, now)
                track.history.append(detection.box.center)
                self.tracks[track.id] = track
                self.next_id += 1
        return list(self.tracks.values())

    def select(self, point: Point, now: float) -> int | None:
        hits = [t for t in self.expire(now) if t.box.contains(point)]
        return min(hits, key=lambda t: t.box.area).id if hits else None
