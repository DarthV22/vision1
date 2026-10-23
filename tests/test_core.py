"""Deterministic measurements, association, expiry and gesture interaction tests."""

from dataclasses import replace
import json
from math import cos, sin
from pathlib import Path
import tempfile
import unittest

from vision1.config import Config
from vision1.geometry import Box, image_point, iou, normalise, speed_direction, velocity
from vision1.hands import GestureControl, Hand, recognise_gesture
from vision1.tracking import Detection, Tracker


def detection(x=0.0, class_id=0, size=100.0):
    return Detection(Box(x, 0, x + size, size), class_id, "object", 0.9)


def hand_points(extended=(True, True, True, True)):
    points = [(0.0, 0.0)] * 21
    points[0] = (0.0, 100.0)
    points[1:5] = [(-30.0, 80.0), (-60.0, 60.0), (-90.0, 40.0), (-120.0, 20.0)]
    for base, x, up in zip((5, 9, 13, 17), (-30, 0, 25, 45), extended):
        points[base] = (float(x), 40.0)
        points[base + 1] = (float(x), 15.0)
        points[base + 2] = (float(x), -10.0 if up else 35.0)
        points[base + 3] = (float(x), -35.0 if up else 60.0)
    return points


class GeometryTests(unittest.TestCase):
    def test_box_geometry_and_degenerate_iou(self):
        box = Box(10, 20, 30, 60)
        self.assertEqual((box.width, box.height, box.center, box.area), (20, 40, (20, 40), 800))
        self.assertAlmostEqual(iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)), 1 / 3)
        self.assertEqual(iou(Box(0, 0, 0, 0), Box(0, 0, 0, 0)), 0)
        self.assertEqual(iou(Box(0, 0, 1, 1), Box(3, 3, 4, 4)), 0)

    def test_velocity_uses_seconds_and_image_direction(self):
        vector = velocity((10, 10), (13, 14), 0.5)
        self.assertEqual(vector, (6, 8))
        speed, direction = speed_direction(vector)
        self.assertEqual(speed, 10)
        self.assertAlmostEqual(direction, 53.13010235)
        self.assertEqual(velocity((0, 0), (1, 1), 0), (0, 0))
        self.assertEqual(speed_direction((0, -1)), (1, 270))

    def test_normalised_coordinates(self):
        self.assertEqual(normalise((320, 180), 640, 360), (0.5, 0.5))
        with self.assertRaises(ValueError):
            normalise((0, 0), 0, 360)

    def test_letterbox_and_pillarbox_clicks(self):
        self.assertEqual(image_point((500, 500), (1000, 1000), (640, 360)), (320, 180))
        self.assertIsNone(image_point((500, 20), (1000, 1000), (640, 360)))
        self.assertEqual(image_point((600, 200), (1200, 400), (400, 400)), (200, 200))
        self.assertIsNone(image_point((20, 200), (1200, 400), (400, 400)))
        self.assertIsNone(image_point((0, 0), (0, 0), (400, 400)))


class TrackingTests(unittest.TestCase):
    def test_persistent_ids_and_measured_velocity(self):
        tracker = Tracker()
        first = tracker.update([detection()], 1.0)[0].id
        track = tracker.update([detection(10)], 1.5)[0]
        self.assertEqual(track.id, first)
        self.assertEqual(track.vector, (20, 0))
        self.assertEqual(track.displacement, (10, 0))
        self.assertEqual(list(track.history), [(50, 50), (60, 50)])

    def test_class_mismatch_creates_new_id(self):
        tracker = Tracker()
        tracker.update([detection()], 1.0)
        tracks = tracker.update([detection(class_id=1)], 1.1)
        self.assertEqual({t.id for t in tracks}, {1, 2})

    def test_one_to_one_association(self):
        tracker = Tracker()
        tracker.update([detection(), detection(120)], 1.0)
        tracks = tracker.update([detection(5), detection(125)], 1.1)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([t.observations for t in tracks], [2, 2])

    def test_expiry_and_brief_occlusion(self):
        tracker = Tracker(max_age=0.8)
        tracker.update([detection()], 1.0)
        self.assertEqual(len(tracker.update([], 1.3)), 1)
        self.assertEqual(tracker.update([detection(5)], 1.5)[0].id, 1)
        self.assertEqual(tracker.expire(2.4), [])
        self.assertEqual(tracker.update([detection()], 2.5)[0].id, 2)

    def test_skipped_inference_is_not_an_observation(self):
        tracker = Tracker()
        tracker.update([detection()], 1.0)
        tracker.expire(1.1)
        tracker.expire(1.2)
        track = tracker.tracks[1]
        self.assertEqual(len(track.history), 1)
        self.assertEqual(track.observations, 1)

    def test_select_smallest_box_and_reject_expired(self):
        tracker = Tracker()
        tracker.update([detection(), detection(25, size=30)], 1.0)
        self.assertEqual(tracker.select((30, 20), 1.1), 2)
        self.assertIsNone(tracker.select((1000, 20), 1.1))
        self.assertIsNone(tracker.select((30, 20), 2.0))

    def test_history_is_bounded(self):
        tracker = Tracker()
        for step in range(100):
            tracker.update([detection(step)], step * 0.1)
        self.assertEqual(len(tracker.tracks[1].history), 60)


class GestureTests(unittest.TestCase):
    def test_four_gestures(self):
        self.assertEqual(recognise_gesture(hand_points()), "Open palm")
        self.assertEqual(recognise_gesture(hand_points((False,) * 4)), "Closed fist")
        self.assertEqual(recognise_gesture(hand_points((True, False, False, False))), "Pointing")
        points = hand_points()
        points[4] = points[8]
        self.assertEqual(recognise_gesture(points), "Pinch")

    def test_rotation_and_scale_invariance(self):
        angle = 1.4
        points = [(2 * (x * cos(angle) - y * sin(angle)) + 50,
                   2 * (x * sin(angle) + y * cos(angle)) - 20) for x, y in hand_points()]
        self.assertEqual(recognise_gesture(points), "Open palm")

    def test_invalid_and_degenerate_hands(self):
        self.assertEqual(recognise_gesture([(0, 0)] * 21), "Unknown")
        with self.assertRaises(ValueError):
            recognise_gesture([])

    def test_dwell_cooldown_and_held_gesture(self):
        control = GestureControl()
        hands = [Hand(hand_points(), "Left", "Pinch")]
        self.assertIsNone(control.update(hands, 0.0))
        self.assertIsNone(control.update(hands, 0.2))
        self.assertEqual(control.update(hands, 0.4)[0], "select")
        self.assertIsNone(control.update(hands, 2.0))
        control.update([], 2.1)
        control.update(hands, 2.2)
        self.assertIsNotNone(control.update(hands, 2.6))

    def test_ambiguous_two_hands_never_act(self):
        control = GestureControl()
        hands = [Hand(hand_points(), "Left", "Open palm")] * 2
        control.update(hands, 0.0)
        self.assertIsNone(control.update(hands, 1.0))


class ConfigTests(unittest.TestCase):
    def test_invalid_types_and_ranges(self):
        for changes in ({"camera": -1}, {"camera": True}, {"confidence": float("nan")},
                        {"inference_size": 417}, {"detection_stride": 0}, {"hands": "yes"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Config(), **changes)

    def test_json_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"camera": 2, "hands": True}), encoding="utf-8")
            self.assertEqual(Config.load(str(path)).camera, 2)
            path.write_text('{"typo": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                Config.load(str(path))


if __name__ == "__main__":
    unittest.main()
