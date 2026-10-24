"""Qt interface. A timer consumes the newest processed image only."""

from dataclasses import replace
from time import monotonic

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from vision1 import __version__
from vision1.config import Config
from vision1.engine import DisplayFrame, VisionWorker
from vision1.geometry import image_point

STYLE = """
QWidget { background: #10171d; color: #dce5eb; font-family: 'Segoe UI', 'Arial'; font-size: 12px; }
QLabel#brand { font-size: 27px; font-weight: 700; letter-spacing: 1px; color: #f4f8fa; }
QLabel#subtitle, QLabel#muted { color: #81929f; }
QLabel#section { color: #8ea2b1; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
QLabel#badge { color: #71d6ca; background: #18312f; border-radius: 5px; padding: 6px 12px; font-size: 11px; }
QFrame#panel { background: #151f27; border: 1px solid #25323d; border-radius: 8px; }
QFrame#panel QLabel, QFrame#panel QCheckBox { background: transparent; }
QFrame#line { background: #293641; max-height: 1px; }
QPushButton { background: #243540; border: 1px solid #344753; border-radius: 5px; padding: 9px 12px; font-weight: 600; }
QPushButton:hover { background: #304651; border-color: #64828e; }
QPushButton#primary { background: #59d4c0; color: #0b2424; border: none; }
QPushButton#primary:hover { background: #81e6d6; }
QPushButton:disabled { background: #202a32; color: #64747e; border-color: #293640; }
QComboBox, QSpinBox { background: #10191f; border: 1px solid #34434f; border-radius: 4px; padding: 6px; min-height: 18px; }
QComboBox QAbstractItemView { background: #18252f; selection-background-color: #31534f; }
QComboBox:disabled, QSpinBox:disabled { color: #687c89; }
QCheckBox { spacing: 9px; padding: 4px 0; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #4d6472; border-radius: 4px; background: #10191f; }
QCheckBox::indicator:checked { background: #59d4c0; border: 1px solid #59d4c0; }
QSlider::groove:horizontal { background: #2a3a45; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #59d4c0; border-radius: 2px; }
QSlider::handle:horizontal { background: #b7efe6; width: 12px; margin: -4px 0; border-radius: 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #121b22; width: 6px; }
QScrollBar::handle:vertical { background: #3a4d5a; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #dce5eb; background: #243540; border: 1px solid #4d6472; padding: 5px; }
"""


def label(value: str, name: str | None = None) -> QLabel:
    result = QLabel(value)
    if name:
        result.setObjectName(name)
    return result


def panel() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)
    return frame, layout


class VideoViewport(QWidget):
    selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.message = "Connect to your scene"
        self.hint = "Choose a camera and start the live stream."
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Click a detected object to select it. Click empty space or press Esc to clear.")

    def set_frame(self, image: QImage) -> None:
        self.image = image
        self.update()

    def clear(self, message: str, hint: str) -> None:
        self.image = QImage()
        self.message, self.hint = message, hint
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a1015"))
        if not self.image.isNull():
            scale = min(self.width() / self.image.width(), self.height() / self.image.height())
            width, height = self.image.width() * scale, self.image.height() * scale
            rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
            painter.drawImage(rect, self.image)
            return
        painter.setPen(QPen(QColor("#14212a"), 1))
        for x in range(0, self.width(), 48):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 48):
            painter.drawLine(0, y, self.width(), y)
        cx, cy = self.width() / 2, self.height() / 2 - 50
        painter.setPen(QPen(QColor("#3d716d"), 1.5))
        painter.drawEllipse(QPointF(cx, cy), 25, 25)
        painter.drawLine(QPointF(cx - 38, cy), QPointF(cx + 38, cy))
        painter.drawLine(QPointF(cx, cy - 38), QPointF(cx, cy + 38))
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        painter.setPen(QColor("#c1d2dd"))
        painter.drawText(QRectF(16, cy + 58, self.width() - 32, 32), Qt.AlignmentFlag.AlignCenter, self.message)
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor("#8296a5"))
        painter.drawText(QRectF(24, cy + 98, self.width() - 48, 50), Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, self.hint)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.image.isNull():
            position = event.position()
            point = image_point(
                (position.x(), position.y()), (self.width(), self.height()),
                (self.image.width(), self.image.height()),
            )
            if point is not None:
                self.selected.emit(point)


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.worker: VisionWorker | None = None
        self._closing = False
        self._had_error = False
        self._last_frame: DisplayFrame | None = None
        self._last_telemetry = 0.0
        self.setWindowTitle("vision1 | Real-Time Computer Vision")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 720)
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 18, 24, 16)
        root.setSpacing(16)
        header = QHBoxLayout()
        header.addWidget(label("vision1", "brand"))
        header.addWidget(label("  /  REAL-TIME VISION WORKSPACE", "subtitle"))
        header.addStretch()
        self.badge = label("OFFLINE", "badge")
        header.addWidget(self.badge)
        header.addWidget(label(f"  v{__version__}", "muted"))
        root.addLayout(header)
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)
        sidebar = QWidget()
        sidebar.setFixedWidth(272)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 6, 0)
        sidebar_layout.setSpacing(12)
        self._build_camera(sidebar_layout)
        self._build_features(sidebar_layout)
        self._build_overlays(sidebar_layout)
        self._build_performance(sidebar_layout)
        sidebar_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(284)
        scroll.setWidget(sidebar)
        body.addWidget(scroll)
        content = QVBoxLayout()
        content.setSpacing(12)
        body.addLayout(content, 1)
        viewport_panel, viewport_layout = panel()
        viewport_layout.setContentsMargins(1, 10, 1, 1)
        viewport_header = QHBoxLayout()
        viewport_header.setContentsMargins(14, 0, 14, 4)
        viewport_header.addWidget(label("LIVE VIEW", "section"))
        viewport_header.addStretch()
        self.frame_info = label("NO SIGNAL", "muted")
        viewport_header.addWidget(self.frame_info)
        viewport_layout.addLayout(viewport_header)
        self.viewport = VideoViewport()
        self.viewport.selected.connect(self._select)
        viewport_layout.addWidget(self.viewport, 1)
        content.addWidget(viewport_panel, 1)
        self._build_telemetry(content)
        self.status_label = label("Ready. Camera preview runs locally; enable inference when you need it.", "muted")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.status_label)
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._refresh)
        self.timer.start()
        self.escape = QShortcut(QKeySequence("Escape"), self)
        self.escape.activated.connect(lambda: self._select(None))
        self.start_shortcut = QShortcut(QKeySequence("Space"), self)
        self.start_shortcut.activated.connect(self._toggle_stream)

    def _build_camera(self, layout: QVBoxLayout) -> None:
        frame, inner = panel()
        inner.addWidget(label("01  /  CAPTURE", "section"))
        form = QFormLayout()
        form.setSpacing(10)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 99)
        self.camera_index.setValue(self.config.camera)
        self.camera_index.setToolTip("OpenCV camera index: usually 0 is built-in; try 1 or 2 for USB cameras.")
        self.resolution = QComboBox()
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        requested = (self.config.width, self.config.height)
        if requested not in resolutions:
            resolutions.append(requested)
        for width, height in resolutions:
            self.resolution.addItem(f"{width} x {height}", (width, height))
        self.resolution.setCurrentIndex(resolutions.index(requested))
        self.resolution.setToolTip("Requested resolution; the actual resolution is shown above the video.")
        form.addRow("Camera", self.camera_index)
        form.addRow("Resolution", self.resolution)
        inner.addLayout(form)
        self.start_button = QPushButton("Start stream")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._toggle_stream)
        inner.addWidget(self.start_button)
        layout.addWidget(frame)

    def _build_features(self, layout: QVBoxLayout) -> None:
        frame, inner = panel()
        inner.addWidget(label("02  /  VISION MODULES", "section"))
        self.toggles: dict[str, QCheckBox] = {}
        for key, title, tip in (
            ("detection", "Object detection + IDs", "YOLO11 nano. Downloads a model on first use. Automatically uses CUDA when available."),
            ("hands", "Hand landmarks", "MediaPipe Tasks, up to two hands. Downloads a model on first use."),
            ("flow", "Optical flow", "Sparse Lucas-Kanade flow; includes motion caused by the camera."),
            ("gesture_control", "Gesture interactions", "Enable hand landmarks. Hold a pinch for 0.35 s to select; open palm to clear. One hand only."),
        ):
            self._add_toggle(inner, key, title, tip)
        confidence_header = QHBoxLayout()
        confidence_header.addWidget(label("Detection confidence", "muted"))
        self.confidence_value = label(f"{self.config.confidence:.0%}")
        confidence_header.addStretch()
        confidence_header.addWidget(self.confidence_value)
        inner.addLayout(confidence_header)
        self.confidence = QSlider(Qt.Orientation.Horizontal)
        self.confidence.setRange(5, 95)
        self.confidence.setValue(round(self.config.confidence * 100))
        self.confidence.valueChanged.connect(self._confidence_changed)
        inner.addWidget(self.confidence)
        note = label("Models load on demand. First use needs internet; cached models work offline.", "muted")
        note.setWordWrap(True)
        inner.addWidget(note)
        layout.addWidget(frame)

    def _build_overlays(self, layout: QVBoxLayout) -> None:
        frame, inner = panel()
        inner.addWidget(label("03  /  VISUAL OVERLAYS", "section"))
        for key, title in (
            ("boxes", "Bounding boxes + labels"), ("trajectories", "Target trajectories"),
            ("vectors", "Motion vectors"), ("grid", "Coordinate grid"),
        ):
            self._add_toggle(inner, key, title, "Applies immediately to the next processed frame.")
        self.clear_button = QPushButton("Clear target  /  Esc")
        self.clear_button.clicked.connect(lambda: self._select(None))
        inner.addWidget(self.clear_button)
        layout.addWidget(frame)

    def _build_performance(self, layout: QVBoxLayout) -> None:
        frame, inner = panel()
        inner.addWidget(label("04  /  PERFORMANCE", "section"))
        form = QFormLayout()
        self.processing_width = QComboBox()
        for width in sorted({640, 960, 1280, self.config.processing_width}):
            self.processing_width.addItem(f"{width} px", width)
        self.processing_width.setCurrentIndex(self.processing_width.findData(self.config.processing_width))
        self.processing_width.setToolTip("Maximum working image width. Coordinates and velocities use this image.")
        self.inference_size = QComboBox()
        for size in sorted({320, 416, 640, self.config.inference_size}):
            self.inference_size.addItem(str(size), size)
        self.inference_size.setCurrentIndex(self.inference_size.findData(self.config.inference_size))
        self.inference_size.setToolTip("YOLO input size. Larger sizes can find smaller objects but require more processing.")
        self.stride = QSpinBox()
        self.stride.setRange(1, 10)
        self.stride.setValue(self.config.detection_stride)
        self.stride.setToolTip("Detect every N processed frames. Boxes hold their last measured position between detections.")
        self.device = QComboBox()
        for device in dict.fromkeys(("auto", "cpu", "cuda:0", "mps", self.config.device)):
            self.device.addItem(device, device)
        self.device.setCurrentText(self.config.device)
        self.device.setToolTip("Auto selects CUDA when available, otherwise CPU. CUDA/MPS require compatible hardware and PyTorch.")
        form.addRow("Working width", self.processing_width)
        form.addRow("YOLO image size", self.inference_size)
        form.addRow("Detect every", self.stride)
        form.addRow("YOLO device", self.device)
        inner.addLayout(form)
        note = label("Capture and performance settings apply when starting a stream.", "muted")
        note.setWordWrap(True)
        inner.addWidget(note)
        layout.addWidget(frame)

    def _add_toggle(self, layout: QVBoxLayout, key: str, title: str, tip: str) -> None:
        checkbox = QCheckBox(title)
        checkbox.setChecked(getattr(self.config, key))
        checkbox.setToolTip(tip)
        self.toggles[key] = checkbox
        checkbox.toggled.connect(lambda checked, name=key: self._option_changed(name, checked))
        layout.addWidget(checkbox)

    def _build_telemetry(self, layout: QVBoxLayout) -> None:
        frame, inner = panel()
        heading = QHBoxLayout()
        heading.addWidget(label("TELEMETRY", "section"))
        heading.addStretch()
        heading.addWidget(label("IMAGE PLANE  /  px + px/s", "muted"))
        inner.addLayout(heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        self.metrics: dict[str, QLabel] = {}
        for col, (key, title) in enumerate((
            ("fps", "PROCESSED FPS"), ("processing", "PROCESSING"),
            ("latency", "FRAME AGE"), ("objects", "TRACKS / HANDS"),
        )):
            grid.addWidget(label(title, "section"), 0, col)
            metric = label("--")
            metric.setStyleSheet("font-size: 23px; color: #e0efed; font-weight: 600;")
            grid.addWidget(metric, 1, col)
            self.metrics[key] = metric
        inner.addLayout(grid)
        line = QFrame()
        line.setObjectName("line")
        line.setFixedHeight(1)
        inner.addWidget(line)
        self.target_label = label("TARGET  /  No target selected. Click a detection in the live view.")
        self.target_label.setWordWrap(True)
        self.target_label.setTextFormat(Qt.TextFormat.PlainText)
        self.target_label.setMinimumHeight(44)
        inner.addWidget(self.target_label)
        self.motion_label = label("FLOW  --     |     GESTURE  --", "muted")
        self.motion_label.setWordWrap(True)
        inner.addWidget(self.motion_label)
        layout.addWidget(frame)

    def _option_changed(self, key: str, value: bool) -> None:
        self.config = replace(self.config, **{key: value})
        if self.worker:
            self.worker.configure(self.config)

    def _confidence_changed(self, value: int) -> None:
        self.confidence_value.setText(f"{value}%")
        self.config = replace(self.config, confidence=value / 100)
        if self.worker:
            self.worker.configure(self.config)

    def _select(self, point) -> None:
        if self.worker:
            self.worker.select(point)

    def _toggle_stream(self) -> None:
        if self._closing or not self.start_button.isEnabled():
            return
        if self.worker is not None:
            self._stop()
            return
        width, height = self.resolution.currentData()
        self.config = replace(
            self.config, camera=self.camera_index.value(), width=width, height=height,
            processing_width=self.processing_width.currentData(),
            inference_size=self.inference_size.currentData(),
            detection_stride=self.stride.value(), device=self.device.currentData(),
        )
        self._had_error = False
        self._last_frame = None
        self._reset_telemetry()
        self.worker = VisionWorker(self.config)
        self.worker.status.connect(self._status)
        self.worker.failed.connect(self._failure)
        self.worker.feature_failed.connect(self._feature_failure)
        self.worker.finished.connect(self._finished)
        self._set_capture_enabled(False)
        self.start_button.setText("Stop stream")
        self.badge.setText("CONNECTING")
        self.viewport.clear("Opening camera", "Checking camera access and waiting for the first frame.")
        self.worker.start()

    def _set_capture_enabled(self, enabled: bool) -> None:
        for widget in (self.camera_index, self.resolution, self.processing_width, self.inference_size, self.stride, self.device):
            widget.setEnabled(enabled)

    def _status(self, message: str) -> None:
        self.status_label.setText(message)

    def _failure(self, message: str) -> None:
        self._had_error = True
        self.status_label.setText(f"Camera / processing error: {message}")

    def _feature_failure(self, feature: str, message: str) -> None:
        self.toggles[feature].setChecked(False)
        self.status_label.setText(f"{feature.capitalize()} disabled: {message}. Check the terminal, then enable it to retry.")

    def _stop(self) -> None:
        if self.worker:
            self.start_button.setEnabled(False)
            self.start_button.setText("Stopping...")
            self.badge.setText("STOPPING")
            self.status_label.setText("Finishing the current operation and releasing the camera...")
            self.worker.requestInterruption()

    def _finished(self) -> None:
        can_restart = True
        if self.worker:
            self.worker.wait()
            can_restart = self.worker.camera_released
            self.worker.deleteLater()
            self.worker = None
        self._set_capture_enabled(can_restart)
        self.start_button.setEnabled(can_restart)
        self.start_button.setText("Start stream" if can_restart else "Restart app to reconnect")
        self.badge.setText("OFFLINE")
        self.frame_info.setText("NO SIGNAL")
        self._last_frame = None
        self._reset_telemetry()
        self.viewport.clear("Stream stopped", "Choose a camera and start a new stream.")
        if not self._had_error:
            self.status_label.setText("Stopped. Camera released.")
        if self._closing:
            self.close()

    def _reset_telemetry(self) -> None:
        for metric in self.metrics.values():
            metric.setText("--")
        self.target_label.setText("TARGET  /  No target selected. Click a detection in the live view.")
        self.motion_label.setText("FLOW  --     |     GESTURE  --")

    def _refresh(self) -> None:
        if not self.worker:
            return
        frame = self.worker.take_frame()
        if frame is not None:
            self._last_frame = frame
            self.viewport.set_frame(frame.image)
        if self._last_frame is None or self._closing:
            return
        now = monotonic()
        if now - self._last_telemetry < 0.15:
            return
        self._last_telemetry = now
        frame = self._last_frame
        data = frame.telemetry
        stale = now - frame.timestamp > 1.5
        if self.start_button.isEnabled():
            self.badge.setText("WAITING / LOADING" if stale else "LIVE")
        self.metrics["fps"].setText("--" if stale or not data["fps"] else f"{data['fps']:.1f}")
        self.metrics["processing"].setText(f"{data['processing_ms']:.0f} ms")
        self.metrics["latency"].setText(f"{(now - frame.timestamp) * 1000:.0f} ms")
        self.metrics["objects"].setText(f"{data['objects']} / {data['hands']}")
        cw, ch = data["capture_size"]
        width, height = data["size"]
        self.frame_info.setText(f"CAM {self.config.camera}  /  {cw} x {ch}  /  WORK {width} x {height}")
        target = data["target"]
        if target:
            x, y = target["center"]
            nx, ny = target["normalised"]
            bw, bh = target["size"]
            measured = target["observations"] > 1
            motion = f"{target['speed']:.1f} px/s  /  {target['direction']:.0f} deg" if measured else "Awaiting second observation"
            age = target["age"] + now - frame.timestamp
            state = "  /  LAST SEEN" if age > 0.25 else ""
            self.target_label.setText(
                f"TARGET #{target['id']:02d}  /  {target['label']}  /  {target['confidence']:.0%}{state}\n"
                f"Center {x:.0f}, {y:.0f} px  [{nx:.3f}, {ny:.3f}]  /  Box {bw:.0f} x {bh:.0f} px  /  {motion}"
            )
        else:
            self.target_label.setText("TARGET  /  No target selected. Click a detection in the live view.")
        flow = f"{data['flow_speed']:.1f} px/s  ({data['flow_points']} points)" if data["flow_speed"] is not None else "--"
        self.motion_label.setText(f"FLOW  {flow}     |     GESTURE  {data['gestures']}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            event.ignore()
            self._closing = True
            self.setEnabled(False)
            self._stop()
        else:
            self.timer.stop()
            event.accept()
