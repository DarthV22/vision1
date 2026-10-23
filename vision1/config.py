"""Small, validated configuration; no environment-specific paths."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    camera: int = 0
    width: int = 1280
    height: int = 720
    processing_width: int = 960
    inference_size: int = 416
    detection_stride: int = 2
    confidence: float = 0.45
    device: str = "cpu"
    model_dir: str = "models"
    detection: bool = False
    hands: bool = False
    flow: bool = False
    grid: bool = False
    boxes: bool = True
    trajectories: bool = True
    vectors: bool = True
    gesture_control: bool = False

    def __post_init__(self) -> None:
        for name in ("camera", "width", "height", "processing_width", "inference_size", "detection_stride"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.camera <= 99:
            raise ValueError("camera must be between 0 and 99")
        if not (160 <= self.width <= 3840 and 120 <= self.height <= 2160):
            raise ValueError("Camera resolution must be between 160x120 and 3840x2160")
        if not 160 <= self.processing_width <= 1920:
            raise ValueError("processing_width must be between 160 and 1920")
        if not 160 <= self.inference_size <= 1280 or self.inference_size % 32:
            raise ValueError("inference_size must be a multiple of 32 between 160 and 1280")
        if not 1 <= self.detection_stride <= 10:
            raise ValueError("detection_stride must be between 1 and 10")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (float, int)):
            raise ValueError("confidence must be numeric")
        if not 0.05 <= self.confidence <= 0.95:
            raise ValueError("confidence must be between 0.05 and 0.95")
        for field in fields(self):
            if field.type is bool and type(getattr(self, field.name)) is not bool:
                raise ValueError(f"{field.name} must be a boolean")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string, e.g. cpu, cuda:0, mps")
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            raise ValueError("model_dir must be a non-empty path")

    @classmethod
    def load(cls, path: str) -> "Config":
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = data.keys() - asdict(cls()).keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**data)
