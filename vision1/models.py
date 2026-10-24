"""Lazy model assets and YOLO adapter. Imports stay off the GUI thread."""

import os
from pathlib import Path
from time import monotonic
from typing import Callable
from urllib.request import Request, urlopen

from vision1.geometry import Box
from vision1.tracking import Detection

MODEL_URLS = {
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def ensure_model(
    name: str, directory: str, cancelled: Callable[[], bool] = lambda: False
) -> Path:
    """Download once, atomically. Socket timeout and total deadline bound network waits."""
    url = MODEL_URLS[name]
    path = Path(directory).expanduser().resolve() / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    deadline = monotonic() + 90
    try:
        request = Request(url, headers={"User-Agent": "vision1/1.0"})
        with urlopen(request, timeout=5) as response, temporary.open("wb") as output:
            size = 0
            while True:
                if cancelled():
                    raise InterruptedError("Model download cancelled")
                if monotonic() > deadline:
                    raise TimeoutError("Model download exceeded 90 seconds")
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 64 * 1024 * 1024:
                    raise ValueError("Model download exceeds size limit")
                output.write(chunk)
            expected = response.headers.get("Content-Length")
            if not size or (expected is not None and size != int(expected)):
                raise OSError("Incomplete model download")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class ObjectDetector:
    def __init__(self, model_path: Path, device: str, size: int) -> None:
        if device == "auto":
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.lower().startswith("cuda"):
            import torch

            if torch.version.cuda is None:
                raise RuntimeError(
                    "CUDA requires a CUDA-enabled PyTorch installation; this environment "
                    "has CPU-only PyTorch. Select cpu, or install a compatible CUDA build "
                    "from pytorch.org/get-started/locally"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "PyTorch cannot access CUDA. Check that a supported NVIDIA GPU "
                    "and driver are available, or select cpu"
                )
        # Assets are managed above; inference needs no online service or probes.
        cache = model_path.parent / ".ultralytics"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.size = size

    def detect(self, frame, confidence: float) -> list[Detection]:
        result = self.model.predict(
            frame, conf=confidence, imgsz=self.size, device=self.device,
            verbose=False, max_det=50,
        )[0]
        if result.boxes is None:
            return []
        # One device transfer instead of synchronising each tensor separately.
        data = result.boxes.data.cpu().numpy()
        return [
            Detection(Box(*map(float, row[:4])), int(row[5]), result.names[int(row[5])], float(row[4]))
            for row in data
        ]
