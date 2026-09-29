# --------------------------------------------
# Unit tests: geometry
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import numpy as np

from rap.geometry import rotated_size, transform


def test_rot90_and_flips():
    a = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out, inv = transform(a, rot=1)                  # 90° clockwise
    assert out.shape == (3, 2, 3) and inv is None
    assert np.array_equal(out, np.rot90(a, -1))
    assert np.array_equal(transform(a, flip_h=True)[0], a[:, ::-1])
    assert np.array_equal(transform(a, flip_v=True)[0], a[::-1])


def test_straighten_expands_canvas_and_marks_corners():
    a = np.full((100, 200, 3), 128, np.uint8)
    out, inv = transform(a, angle=10)
    w, h = rotated_size(200, 100, 10)
    assert out.shape[:2] == (round(h), round(w))
    assert inv is not None and inv[0, 0] and not inv[out.shape[0] // 2, out.shape[1] // 2]


def test_normalised_crop():
    a = np.zeros((100, 200, 3), np.uint8)
    out, _ = transform(a, crop=(0.25, 0.5, 0.75, 1.0))
    assert out.shape[:2] == (50, 100)
