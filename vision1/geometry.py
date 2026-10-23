"""Image-plane mathematics. Origin is top left; +x right, +y down."""

from dataclasses import dataclass
from math import atan2, degrees, hypot

Point = tuple[float, float]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, point: Point) -> bool:
        return self.x1 <= point[0] <= self.x2 and self.y1 <= point[1] <= self.y2

    def translated(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


def iou(a: Box, b: Box) -> float:
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def normalise(point: Point, width: int, height: int) -> Point:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return point[0] / width, point[1] / height


def velocity(previous: Point, current: Point, dt: float) -> Point:
    if dt <= 0:
        return (0.0, 0.0)
    return (current[0] - previous[0]) / dt, (current[1] - previous[1]) / dt


def speed_direction(vector: Point) -> Point:
    """Return pixels/s and clockwise degrees from image-right."""
    speed = hypot(*vector)
    return speed, degrees(atan2(vector[1], vector[0])) % 360 if speed else 0.0


def image_point(
    point: Point, widget_size: tuple[int, int], image_size: tuple[int, int]
) -> Point | None:
    """Invert an aspect-fit viewport, rejecting clicks on letterbox margins."""
    ww, wh = widget_size
    iw, ih = image_size
    if min(ww, wh, iw, ih) <= 0:
        return None
    scale = min(ww / iw, wh / ih)
    x = (point[0] - (ww - iw * scale) / 2) / scale
    y = (point[1] - (wh - ih * scale) / 2) / scale
    return (x, y) if 0 <= x < iw and 0 <= y < ih else None
