"""Processing worker: capture -> inference -> measurements -> overlay -> mailbox."""

from dataclasses import dataclass
import logging
from threading import Lock
from time import monotonic

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from vision1.camera import Camera
from vision1.config import Config
from vision1.geometry import normalise, speed_direction
from vision1.hands import GestureControl, HandProcessor
from vision1.models import ObjectDetector, ensure_model
from vision1.motion import FlowResult, OpticalFlow
from vision1.overlays import render, text
from vision1.tracking import Tracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DisplayFrame:
    image: QImage
    telemetry: dict
    timestamp: float


class VisionWorker(QThread):
    status = Signal(str)
    failed = Signal(str)
    feature_failed = Signal(str, str)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._lock = Lock()
        self._config = config
        self._latest: DisplayFrame | None = None
        self._selection: tuple[float, float] | None = None
        self._selection_pending = False
        self.camera_released = True

    def configure(self, config: Config) -> None:
        with self._lock:
            self._config = config

    def select(self, point: tuple[float, float] | None) -> None:
        with self._lock:
            self._selection = point
            self._selection_pending = True

    def take_frame(self) -> DisplayFrame | None:
        with self._lock:
            frame, self._latest = self._latest, None
            return frame

    def _settings(self) -> tuple[Config, bool, tuple[float, float] | None]:
        with self._lock:
            result = self._config, self._selection_pending, self._selection
            self._selection_pending = False
            return result

    def run(self) -> None:
        with self._lock:
            config = self._config
        camera = Camera(config.camera, config.width, config.height)
        detector = None
        hand_processor = None
        blocked: set[str] = set()
        tracker, flow_processor, gestures = Tracker(), OpticalFlow(), GestureControl()
        selected = None
        sequence, frame_number = 0, 0
        previous_time = None
        fps = 0.0
        last_shape = None
        last_detection_enabled = False
        last_gesture_enabled = False
        try:
            self.status.emit(f"Opening camera {config.camera}...")
            camera.start()
            self.status.emit("Waiting for camera frames...")
            while not self.isInterruptionRequested():
                packet = camera.latest(sequence)
                if packet is None:
                    continue
                sequence = packet.sequence
                config, selection_pending, selection = self._settings()
                # An explicit off/on cycle permits retry after a feature failure.
                for name in ("detection", "hands", "flow"):
                    if not getattr(config, name):
                        blocked.discard(name)
                started = monotonic()
                frame = packet.image
                if frame.shape[1] > config.processing_width:
                    scale = config.processing_width / frame.shape[1]
                    frame = cv2.resize(frame, (config.processing_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                else:
                    frame = frame.copy()
                height, width = frame.shape[:2]
                if last_shape != (height, width):
                    tracker = Tracker()
                    flow_processor.reset()
                    selected = None
                    last_shape = (height, width)
                hands = []
                flow = FlowResult()
                detection_enabled = config.detection and "detection" not in blocked
                if not detection_enabled:
                    if last_detection_enabled:
                        tracker = Tracker()
                        selected = None
                    tracks = []
                else:
                    try:
                        if detector is None:
                            self.status.emit("Loading YOLO11 nano; downloading model if missing...")
                            path = ensure_model("yolo11n.pt", config.model_dir, self.isInterruptionRequested)
                            if self.isInterruptionRequested():
                                break
                            detector = ObjectDetector(path, config.device, config.inference_size)
                            self.status.emit(f"YOLO ready on {detector.device} | click a detection to select a target")
                        if frame_number % config.detection_stride == 0 or not last_detection_enabled:
                            tracks = tracker.update(detector.detect(frame, config.confidence), packet.timestamp)
                        else:
                            tracks = tracker.expire(packet.timestamp)
                    except InterruptedError:
                        break
                    except Exception as exc:
                        log.exception("Object detection failed")
                        blocked.add("detection")
                        detector = None
                        tracker = Tracker()
                        tracks, selected = [], None
                        self.feature_failed.emit("detection", str(exc))
                last_detection_enabled = detection_enabled
                if config.hands and "hands" not in blocked:
                    try:
                        if hand_processor is None:
                            self.status.emit("Loading hand landmarks; downloading model if missing...")
                            path = ensure_model("hand_landmarker.task", config.model_dir, self.isInterruptionRequested)
                            if self.isInterruptionRequested():
                                break
                            hand_processor = HandProcessor(path)
                            self.status.emit("Hand landmarks ready | gesture interactions are optional")
                        hands = hand_processor.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), packet.timestamp)
                    except InterruptedError:
                        break
                    except Exception as exc:
                        log.exception("Hand processing failed")
                        blocked.add("hands")
                        if hand_processor is not None:
                            hand_processor.close()
                            hand_processor = None
                        self.feature_failed.emit("hands", str(exc))
                elif hand_processor is not None:
                    hand_processor.close()
                    hand_processor = None
                if config.flow and "flow" not in blocked:
                    try:
                        flow = flow_processor.update(frame, packet.timestamp)
                    except Exception as exc:
                        log.exception("Optical flow failed")
                        blocked.add("flow")
                        flow_processor.reset()
                        self.feature_failed.emit("flow", str(exc))
                else:
                    flow_processor.reset()
                if selection_pending:
                    selected = tracker.select(selection, packet.timestamp) if selection else None
                gesture_enabled = config.gesture_control and config.hands
                if gesture_enabled != last_gesture_enabled:
                    gestures = GestureControl()
                if gesture_enabled:
                    action = gestures.update(hands, packet.timestamp)
                    if action:
                        selected = tracker.select(action[1], packet.timestamp) if action[0] == "select" else None
                last_gesture_enabled = gesture_enabled
                if selected not in tracker.tracks:
                    selected = None
                render(frame, tracks, hands, flow, selected, config, packet.timestamp)
                now = monotonic()
                if previous_time is not None and now > previous_time:
                    instant = 1.0 / (now - previous_time)
                    fps = instant if not fps else 0.85 * fps + 0.15 * instant
                previous_time = now
                target = tracker.tracks.get(selected)
                target_data = None
                if target:
                    speed, direction = speed_direction(target.vector)
                    target_data = {
                        "id": target.id, "label": target.detection.label,
                        "confidence": target.detection.confidence, "center": target.box.center,
                        "normalised": normalise(target.box.center, width, height),
                        "size": (target.box.width, target.box.height), "speed": speed,
                        "direction": direction, "vector": target.vector,
                        "displacement": target.displacement, "age": packet.timestamp - target.last_seen,
                        "observations": target.observations,
                    }
                processing_ms = (monotonic() - started) * 1000
                fps_text = f"{fps:.1f}" if fps else "--"
                text(frame, f"{fps_text} FPS | {processing_ms:.1f} ms processing", (14, 24))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                telemetry = {
                    "fps": fps, "processing_ms": processing_ms,
                    "latency_ms": (monotonic() - packet.timestamp) * 1000,
                    "objects": len(tracks), "hands": len(hands),
                    "gestures": ", ".join(hand.gesture for hand in hands) or "--",
                    "target": target_data, "size": (width, height),
                    "capture_size": (packet.image.shape[1], packet.image.shape[0]),
                    "sequence": sequence, "flow_points": len(flow.vectors),
                    "flow_speed": speed_direction(flow.velocity)[0] if flow.valid else None,
                }
                with self._lock:
                    self._latest = DisplayFrame(image, telemetry, packet.timestamp)
                if frame_number == 0 and not blocked:
                    self.status.emit("Live | click an object to select a target")
                frame_number += 1
        except Exception as exc:
            log.exception("Vision worker stopped")
            self.failed.emit(str(exc))
        finally:
            if hand_processor is not None:
                try:
                    hand_processor.close()
                except Exception:
                    log.exception("Hand processor cleanup failed")
            self.camera_released = camera.stop()
            if not self.camera_released:
                log.warning("Camera driver did not stop within one second; capture thread is a daemon")
                self.failed.emit("Camera driver is unresponsive. Restart the app before reopening this camera.")
