"""Run with python -m vision1."""

import argparse
import logging
import sys

from vision1.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description="vision1 | Real-time computer vision")
    parser.add_argument("--config", help="Path to a JSON configuration file")
    parser.add_argument("--download-models", action="store_true", help="Cache models, then exit")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = Config.load(args.config) if args.config else Config()
        if args.download_models:
            from vision1.models import ensure_model

            for name in ("yolo11n.pt", "hand_landmarker.task"):
                print(ensure_model(name, config.model_dir))
            return 0
        from PySide6.QtWidgets import QApplication
        from vision1.gui import MainWindow

        app = QApplication(sys.argv[:1])
        app.setApplicationName("vision1")
        app.setOrganizationName("vision1")
        window = MainWindow(config)
        window.show()
        return app.exec()
    except Exception as exc:
        logging.getLogger("vision1").error("Unable to start: %s", exc, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
