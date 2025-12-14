import numpy as np

from birdid.utils.roi import clamp_roi, pad_roi, roi_area, safe_roi_from_center


def test_clamp_negative_coords():
    roi = clamp_roi((-10, -5, 5, 5), 100, 100)
    assert roi == (0.0, 0.0, 5.0, 5.0)


def test_clamp_overflow_coords():
    roi = clamp_roi((90, 90, 150, 200), 100, 120)
    assert roi[0] == 90.0 and roi[1] == 90.0
    assert roi[2] == 99.0 and roi[3] == 119.0


def test_safe_roi_area_and_padding():
    roi = safe_roi_from_center(50, 50, 20, 10, 100, 100)
    padded = clamp_roi(pad_roi(roi, 5), 100, 100)
    assert roi_area(padded) > roi_area(roi)
