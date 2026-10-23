# vision1

[![Tests](https://github.com/DarthV22/vision1/actions/workflows/tests.yml/badge.svg)](https://github.com/DarthV22/vision1/actions/workflows/tests.yml)

A local computer vision desktop workspace built with Python, OpenCV and PySide6. Watch a live webcam, identify common objects with YOLO11 nano, track targets, inspect hand landmarks and visualise image-plane motion.

The dark interface puts the scene first: an aspect-preserving viewport, compact controls, selectable targets and telemetry calculated from processed frames. No cloud inference, recording or fabricated demo measurements.

## Install and run

Use a 64-bit Python installation with wheels available for PyTorch, MediaPipe and PySide6. Python 3.12 or 3.13 is a practical starting point. A webcam and desktop session are needed for live use.

Windows PowerShell, from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m vision1
```

macOS / Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m vision1
```

PySide6 Essentials supplies the Qt modules used here without Qt WebEngine or Addons. PyTorch is a transitive dependency of Ultralytics; the model is lightweight, but the complete Python environment is larger. Ultralytics and MediaPipe request separate OpenCV distributions; both are constrained to OpenCV 4 here.

In VS Code, open this folder, install the Python and Python Debugger extensions, run **Python: Select Interpreter**, choose `.venv`, then press **F5** with the included `vision1` launch configuration. The terminal commands also work without VS Code.

## Use

1. Choose a camera index and requested resolution, then **Start stream**. Index `0` is commonly the built-in/default webcam; try `1` or `2` for another camera. Devices can negotiate a different resolution; the header reports the actual capture and working sizes.
2. Enable **Object detection + IDs**, **Hand landmarks** or **Optical flow** as needed. All are off initially for a fast preview. First use downloads the relevant model to `models/`; subsequent launches use the cached file. The interface stays responsive while models load, although processed video pauses during loading.
3. Click a detected object to select it. The target receives a gold box and crosshair; telemetry shows its ID, confidence, pixel/normalised centre, dimensions, speed and direction. Click empty image space, press **Esc**, or use **Clear target** to clear it. Clicks in letterbox margins are ignored.
4. Enable **Gesture interactions** together with **Hand landmarks** for gesture control. With exactly one visible hand, hold a pinch for 0.35 seconds to select the object at the thumb/index midpoint; hold an open palm to clear. A one-second cooldown and edge trigger prevent repeated actions. Pointing and closed fist are recognised and displayed. These are geometric heuristics, sensitive to occlusion and viewing angle.
5. Toggle boxes, trajectories, vectors and the pixel grid independently. Confidence changes take effect live. Capture and performance settings apply at the next start. **Space** starts/stops the stream when a focused control does not consume it.

Stop and close wait for the current inference operation to finish, then release resources without blocking Qt's event loop. A disconnected camera stops the stream with a visible error. A failed inference feature disables itself while preview continues; check the terminal and toggle the feature to retry. A camera driver stuck inside an operating-system call cannot be forcibly interrupted safely: its daemon capture thread is abandoned after a bounded wait, and the app reports that a restart is required.

On Windows, capture prefers DirectShow because the default Media Foundation backend can open some HP cameras without returning frames. Other platforms use OpenCV's native default; Windows also falls back to that default if DirectShow cannot open the device.

To cache both models before offline use:

```sh
python -m vision1 --download-models
```

Downloads use HTTPS, a socket timeout, a total deadline, a size limit and atomic file replacement. For an offline machine, place `yolo11n.pt` and `hand_landmarker.task` from the official model links below in the configured model directory. An invalid cached file produces a visible feature error; remove it and retry. Frames stay on the local machine; package installation and missing model files require internet. Ultralytics online probes are disabled by default. MediaPipe Tasks may attempt to send API performance/utilisation metrics, as described in its [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice); its inference also works without a connection.

## Configuration and performance

```sh
python -m vision1 --config config.example.json
python -m vision1 --debug
```

The example lists all supported keys. Unknown keys, wrong types and invalid ranges are rejected. Paths are relative to the working directory. UI settings last for the current process; edit your JSON file for persistent defaults.

CPU is the default. Set the YOLO device to `cuda:0` or `mps` only with compatible hardware and an appropriate PyTorch installation. Hand landmarks and optical flow use the CPU. An unsupported device produces a feature error rather than silently reporting GPU acceleration.

| Performance control | Effect |
| --- | --- |
| Working width | Maximum width used by the whole processing pipeline. Lower values reduce work and image detail; all pixel coordinates and speeds refer to this resized image. Smaller camera images are never enlarged. |
| YOLO image size | Detector input size: 320 is faster, 416 is the default, 640 retains more detail for small objects at greater cost. This is independent of the working width. |
| Detect every | Run detection once per N processed frames. 1 updates every frame; 2 halves the detection frequency. Boxes hold their last measured positions between detections; larger values can lose IDs during movement. Hands and flow still run each processed frame. |
| YOLO device | `cpu` uses the processor; `cuda:0` requests the first NVIDIA GPU; `mps` requests Apple's Metal backend on compatible Macs. This affects YOLO only; ID association runs on the CPU. |

Stop the stream before changing these controls, then start it again. Camera resolution controls capture detail and bandwidth before the working resize. Detection confidence is a live threshold: higher values reject more uncertain detections, while lower values can increase false positives. Disabling unused vision modules saves processing; hiding overlays only reduces drawing work.

To diagnose CUDA in the same environment used to launch the app:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
```

A version ending in `+cpu` with `torch.version.cuda` equal to `None` means PyTorch cannot run CUDA even when an NVIDIA GPU is installed. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a CUDA build compatible with your Python and driver, installing through the same environment's Python. Restart the app after installation. Selecting `cuda:0` does not install GPU support. Detection and IDs share one feature toggle, so a detector device error disables both; this does not mean the ID tracker requires CUDA.

For weaker machines, choose a 640-pixel working width, a 320-pixel YOLO size, detect every 2–3 processed frames, and enable only needed modules. The working width is a maximum: small images are never enlarged. More frame skipping holds measured boxes longer and can reduce tracking quality. Model imports and assets are loaded on demand; capture and output each retain only one frame. UI presentation polls at approximately 33 Hz and telemetry at approximately 7 Hz, independent of inference speed.

## Architecture

```text
OpenCV capture thread -> latest CameraFrame -> vision QThread -> latest DisplayFrame
                                                ^                     |
                                                |                 Qt GUI timer
                                           settings / clicks          |
                                                               viewport + telemetry
```

| Module | Responsibility |
| --- | --- |
| `camera.py` | Acquisition, monotonic timestamps, one-frame mailbox, disconnects |
| `engine.py` | Worker lifecycle, feature orchestration, bounded output mailbox |
| `models.py` | Asset cache and lazy Ultralytics adapter |
| `tracking.py` | Class-aware association, persistent IDs, trajectory history, expiry |
| `hands.py` | MediaPipe Tasks VIDEO API, gesture geometry, interaction debounce |
| `motion.py` | Sparse Lucas–Kanade flow and forward/backward rejection |
| `geometry.py` | Bounding boxes, coordinates, velocity, viewport mapping |
| `overlays.py` | OpenCV rendering from processed observations |
| `gui.py` | Qt controls, aspect-fit display, telemetry, asynchronous stop/close |
| `config.py` | Validated JSON configuration |

There are two application background threads: capture and processing. Libraries may use their own native threads. Capture owns its `VideoCapture`; processing owns its models and tracker; Qt owns widgets. Locks protect settings and mailbox swaps. No per-frame Qt signal queue accumulates when inference or rendering falls behind. Images crossing into Qt own their pixel storage.

## Mathematics and measurement limits

- **Coordinates:** image origin is top-left, `+x` points right, `+y` down. Normalised coordinates are `(x/W, y/H)` in the **working image**, not the original sensor. Selection inverts the viewport's aspect-fit scale and offsets.
- **Boxes:** width `x2-x1`, height `y2-y1`, centre `((x1+x2)/2, (y1+y2)/2)`. Association scores use intersection-over-union `area(A intersect B) / area(A union B)`, with zero for a degenerate union.
- **Tracking:** detections are associated greedily, one-to-one, within a class using IoU and a short-distance gate. A constant-velocity prediction guides association; displayed boxes remain the last measured observation. Tracks survive up to 0.8 seconds without a detection. IDs persist through brief misses, but are not appearance-based identities: crossings, long occlusions, fast movement or sparse inference can change IDs. Restarting or disabling detection resets tracking.
- **Object motion:** displacement is `c_now-c_previous`; velocity is displacement divided by elapsed **capture time in seconds** between matched detector observations. Speed is `sqrt(vx²+vy²)` in px/s. Direction is `atan2(vy,vx)`, clockwise from image-right; an angle at zero speed is only a convention. No speed is displayed until two observations exist. Inference-skipped frames add no synthetic observations. Histories retain at most 60 measured centres per track.
- **Optical flow:** pyramidal Lucas–Kanade estimates feature displacement under local brightness constancy, approximately `Ix*u + Iy*v + It = 0`. A forward/backward consistency check rejects unstable points. Median displacement divided by frame time gives the reported **scene** flow speed. Flow includes camera movement and is independent of object tracking. Object arrows represent 0.15 seconds of measured velocity, capped at 100 pixels; flow arrows show measured frame-to-frame displacement.
- **Timing:** processed FPS is an exponentially averaged reciprocal of output intervals. Processing time includes resize, inference, measurements and overlays up to the timing annotation. Frame age measures time since OpenCV returned a captured frame until the GUI reads telemetry; it includes waiting, processing and presentation scheduling, but cannot measure buffering inside the camera/driver. A stalled image is marked waiting/loading and FPS becomes unavailable.
- **Projection and calibration:** a pinhole camera follows `s*[u,v,1]^T = K*[R|t]*[X,Y,Z,1]^T`. Intrinsics `K` describe focal lengths and principal point; lens distortion must be estimated separately. OpenCV `calibrateCamera` and `undistort` can support a future calibration workflow. A pixel displacement alone gives no metres, depth or physical velocity: those require calibration plus depth or a known scene plane. This app reports image-plane measurements only. Kalman smoothing is intentionally optional and not implemented; direct measurements remain easy to inspect.

## Tests

```sh
python -m unittest discover -s tests -v
python -m compileall -q vision1 tests
python -m pip check
```

The tests cover geometry, time-based velocity, viewport mapping, class-aware one-to-one association, missed detections, expiry, bounded histories, gestures and debounce, synthetic optical flow, a fake camera, offscreen Qt rendering and worker/GUI shutdown. They use no camera or network and do not download models. Webcam permissions, driver behaviour, live gesture quality, GPU support and real-world frame rates require testing on your hardware.

GitHub Actions runs these checks on Windows, Linux and macOS with Python 3.12. The repository contains source, tests and configuration; virtual environments, model weights, caches and generated output remain local and are excluded from Git.

Development verification on Windows with Python 3.14.6: all 28 tests, Qt Essentials 6.10.3 desktop rendering, compilation and `pip check` passed. Separate real-model CPU checks detected a bus and people in Ultralytics' bundled sample and processed a blank frame with MediaPipe. The complete worker also processed live HP webcam frames at 1280x720 capture resolution with detection, hand landmarks and optical flow enabled, then stopped and released the camera cleanly. GPU acceleration remains unverified.

If the camera will not open, close other camera applications and check OS camera permissions. On Linux, check access to `/dev/video*`; on Ubuntu/Debian, install Qt's graphics libraries with `sudo apt-get install libegl1 libopengl0` if they are missing. Desktop X11 sessions may also need XCB libraries. If `python` opens the Windows Store, install Python or select its actual interpreter in VS Code. If a package has no wheel for your platform/Python combination, use a supported 64-bit Python release rather than attempting to compile the entire stack.

## Upstream references

- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/) and [YOLO11 nano weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) and [hand model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html), [camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) and [Qt for Python](https://doc.qt.io/qtforpython-6/)

Upstream dependencies and model weights retain their own licences. Review the [Ultralytics repository's licence](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) before distributing a project using its code or weights.
