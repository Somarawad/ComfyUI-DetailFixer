"""Run with: python -m pytest tests  (no ComfyUI needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detail_core as c  # noqa: E402


def test_manual_pixels_and_percent():
    b = c.parse_manual_boxes("10, 20, 30, 40\n50%, 50%, 10%, 10%", 200, 100)
    assert b[0]["x0"] == 10 and b[0]["y1"] == 60
    assert b[1]["x0"] == 100 and b[1]["y0"] == 50 and b[1]["x1"] == 120


def test_clamp():
    b = c.clamp_box({"x0": -5, "y0": -5, "x1": 500, "y1": 500}, 100, 80)
    assert (b["x0"], b["y0"], b["x1"], b["y1"]) == (0, 0, 100, 80)


def test_gemini_boxes_are_ymin_xmin_ymax_xmax_per_mille():
    b = c.parse_gemini_boxes('[{"box_2d": [100, 200, 300, 400], "label": "logo"}]', 1000, 500)
    assert (b[0]["x0"], b[0]["y0"], b[0]["x1"], b[0]["y1"]) == (200, 50, 400, 150)


def test_sam3_boxes_merge_and_dedupe():
    a = [[{"x": 484.0, "y": 432.25, "width": 73.5, "height": 18.0, "score": 0.87}]]
    b = [[{"x": 314, "y": 520, "width": 237.5, "height": 69, "score": 0.9},
          {"x": 485, "y": 432, "width": 73, "height": 18, "score": 0.5}]]
    boxes = c.parse_bounding_boxes([a, b], 896, 1152)
    assert len(boxes) == 2 and boxes[0]["x0"] == 314


def test_crop_region_contains_box_and_is_multiple_of_16():
    box = {"x0": 480, "y0": 430, "x1": 560, "y1": 450, "label": "logo"}
    r = c.crop_region(box, 896, 1152)
    x0, y0, x1, y1 = r["crop"]
    assert x0 <= 480 and y0 <= 430 and x1 >= 560 and y1 >= 450
    assert r["size"][0] % 16 == 0 and r["size"][1] % 16 == 0


def test_merge_touching_wordmark_and_tagline():
    boxes = [{"label": "a", "x0": 658, "y0": 551, "x1": 952, "y1": 679},
             {"label": "b", "x0": 727, "y0": 653, "x1": 846, "y1": 703},
             {"label": "c", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]
    m = c.merge_touching_boxes(boxes)
    assert len(m) == 2
    assert (m[0]["x0"], m[0]["y0"], m[0]["x1"], m[0]["y1"]) == (658, 551, 952, 703)
