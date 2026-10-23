"""Synthetic frames and a fake camera; no webcam, downloads or model weights."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
from time import monotonic, sleep
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from vision1.camera import Camera
from vision1.config import Config
from vision1.engine import VisionWorker
from vision1.gui import MainWindow
from vision1.motion import OpticalFlow
from vision1.models import ObjectDetector
from vision1.geometry import Box
from vision1.hands import Hand
from vision1.tracking import Detection


class FakeCapture:
    instances = []

    def __init__(self, index, backend=None):
        self.released = False
        self.index = index
        self.backend = backend
        self.instances.append(self)

    def isOpened(self):
        return self.index != 99

    def set(self, *_):
        return True

    def read(self):
        sleep(0.005)
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, *_):
        pass

    def detect(self, frame, confidence):
        return [Detection(Box(20, 30, 120, 130), 0, "person", 0.9)]


class FakeHands:
    def __init__(self, *_):
        pass

    def detect(self, rgb, timestamp):
        return [Hand([(150 + i, 50 + i) for i in range(21)], "Left", "Unknown")]

    def close(self):
        pass


class DeviceTests(unittest.TestCase):
    def test_cpu_only_torch_reports_actionable_cuda_error(self):
        torch = SimpleNamespace(version=SimpleNamespace(cuda=None))
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                ObjectDetector(Path("unused.pt"), "cuda:0", 320)

    def test_unavailable_cuda_reports_driver_error(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU and driver"):
                ObjectDetector(Path("unused.pt"), "cuda:0", 320)


class MotionTests(unittest.TestCase):
    def test_known_image_translation(self):
        rng = np.random.default_rng(12)
        gray = rng.integers(0, 256, (180, 240), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 2], [0, 1, -1]]), (240, 180))
        flow = OpticalFlow()
        self.assertFalse(flow.update(frame, 1.0).valid)
        result = flow.update(moved, 1.05)
        self.assertTrue(result.valid)
        self.assertGreater(len(result.vectors), 10)
        self.assertAlmostEqual(result.velocity[0], 40, delta=3)
        self.assertAlmostEqual(result.velocity[1], -20, delta=3)

    def test_blank_and_changed_resolution(self):
        flow = OpticalFlow()
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        self.assertFalse(flow.update(frame, 1.0).valid)
        self.assertFalse(flow.update(frame, 1.1).valid)
        self.assertFalse(flow.update(np.zeros((120, 160, 3), dtype=np.uint8), 1.2).valid)


class CameraTests(unittest.TestCase):
    @patch("vision1.camera.sys.platform", "win32")
    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    def test_windows_uses_directshow(self):
        camera = Camera(0, 320, 240)
        camera.start()
        try:
            self.assertIsNotNone(camera.latest(0, timeout=1))
            self.assertEqual(FakeCapture.instances[-1].backend, cv2.CAP_DSHOW)
        finally:
            camera.stop()

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    def test_latest_frame_and_release(self):
        camera = Camera(0, 320, 240)
        camera.start()
        try:
            first = camera.latest(0, timeout=1)
            self.assertIsNotNone(first)
            sleep(0.06)
            latest = camera.latest(first.sequence)
            self.assertGreater(latest.sequence, first.sequence + 1)
        finally:
            self.assertTrue(camera.stop())
        self.assertTrue(FakeCapture.instances[-1].released)

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    def test_unavailable_camera_is_reported(self):
        camera = Camera(99, 320, 240)
        with self.assertLogs("vision1.camera", level="ERROR"):
            camera.start()
            try:
                with self.assertRaisesRegex(OSError, "unavailable"):
                    camera.latest(0, timeout=1)
            finally:
                camera.stop()


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    def test_preview_worker_mailbox_and_clean_stop(self):
        worker = VisionWorker(Config())
        worker.start()
        packet = None
        try:
            deadline = monotonic() + 3
            while packet is None and monotonic() < deadline:
                self.app.processEvents()
                packet = worker.take_frame()
                sleep(0.01)
            self.assertIsNotNone(packet)
            self.assertEqual(packet.telemetry["size"], (320, 240))
            self.assertEqual(packet.telemetry["objects"], 0)
            self.assertFalse(packet.image.isNull())
        finally:
            worker.requestInterruption()
            self.assertTrue(worker.wait(3000))
        worker.take_frame()  # Drain any final output now that the producer stopped.
        self.assertIsNone(worker.take_frame())

    def test_window_paint_configuration_and_shutdown(self):
        window = MainWindow(Config())
        try:
            window.show()
            self.app.processEvents()
            self.assertEqual(window.camera_index.value(), 0)
            self.assertEqual(window.resolution.currentData(), (1280, 720))
            window.toggles["grid"].setChecked(True)
            self.assertTrue(window.config.grid)
            window.confidence.setValue(60)
            self.assertEqual(window.config.confidence, 0.6)
            self.assertFalse(window.grab().isNull())
        finally:
            window.close()

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    @patch("vision1.engine.ensure_model", return_value=Path("unused"))
    @patch("vision1.engine.ObjectDetector", FakeDetector)
    @patch("vision1.engine.HandProcessor", FakeHands)
    def test_inference_overlays_and_target_telemetry(self, _model):
        worker = VisionWorker(replace(Config(), detection=True, hands=True, grid=True))
        worker.select((50, 60))
        worker.start()
        try:
            packet = None
            deadline = monotonic() + 3
            while monotonic() < deadline:
                packet = worker.take_frame()
                if packet and packet.telemetry["target"] and packet.telemetry["target"]["observations"] > 1:
                    break
                self.app.processEvents()
                sleep(0.01)
            self.assertIsNotNone(packet)
            self.assertEqual(packet.telemetry["objects"], 1)
            self.assertEqual(packet.telemetry["hands"], 1)
            target = packet.telemetry["target"]
            self.assertEqual(target["center"], (70, 80))
            self.assertEqual(target["size"], (100, 100))
            self.assertEqual(target["speed"], 0)
            self.assertEqual(target["id"], 1)
        finally:
            worker.requestInterruption()
            self.assertTrue(worker.wait(3000))

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    @patch("vision1.engine.ensure_model", side_effect=OSError("offline"))
    def test_failed_model_leaves_camera_preview_running(self, _model):
        window = MainWindow(replace(Config(), detection=True))
        window.show()
        try:
            with self.assertLogs("vision1.engine", level="ERROR"):
                window._toggle_stream()
                deadline = monotonic() + 3
                while (window.config.detection or window._last_frame is None) and monotonic() < deadline:
                    self.app.processEvents()
                    sleep(0.01)
            self.assertFalse(window.config.detection)
            self.assertIsNotNone(window._last_frame)
            self.assertTrue(window.worker.isRunning())
            self.assertIn("offline", window.status_label.text())
        finally:
            window.close()
            deadline = monotonic() + 3
            while window.worker is not None and monotonic() < deadline:
                self.app.processEvents()
                sleep(0.01)
            if window.worker:
                window.worker.requestInterruption()
                window.worker.wait(3000)
                self.app.processEvents()
            window.close()

    @patch("vision1.camera.cv2.VideoCapture", FakeCapture)
    def test_full_gui_start_stop_and_deferred_close(self):
        window = MainWindow(replace(Config(), flow=True))
        window.show()
        window._toggle_stream()
        try:
            deadline = monotonic() + 3
            while window._last_frame is None and monotonic() < deadline:
                self.app.processEvents()
                sleep(0.01)
            self.assertIsNotNone(window._last_frame)
            self.assertFalse(window.camera_index.isEnabled())
            window.close()
            deadline = monotonic() + 3
            while window.worker is not None and monotonic() < deadline:
                self.app.processEvents()
                sleep(0.01)
            self.assertIsNone(window.worker)
            self.assertFalse(window.isVisible())
        finally:
            if window.worker:
                window.worker.requestInterruption()
                window.worker.wait(3000)
                self.app.processEvents()
            window.close()


if __name__ == "__main__":
    unittest.main()
