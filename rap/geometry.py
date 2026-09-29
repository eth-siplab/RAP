# --------------------------------------------
# Crop, rotate, straighten and flip
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Crop / rotate / straighten / flip, applied to the full-resolution image.

Order: flip, quarter turns (clockwise), straighten by `angle` degrees
(clockwise, canvas expanded to the rotated bounding box), then crop with a
rectangle given in normalised coordinates of that rotated frame. The browser
preview (app.js, crop mode) uses the same order and conventions.
"""
import math

import cv2
import numpy as np


def rotated_size(w, h, angle):
    a = math.radians(angle)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return w * c + h * s, w * s + h * c


def transform(img, rot=0, flip_h=False, flip_v=False, angle=0.0, crop=None):
    """Returns (image, invalid mask or None). invalid marks pixels outside the source."""
    if flip_h:
        img = img[:, ::-1]
    if flip_v:
        img = img[::-1]
    img = np.ascontiguousarray(np.rot90(img, -int(rot) % 4))   # np.rot90 turns counter-clockwise
    invalid = None
    if abs(angle) > 1e-3:
        H, W = img.shape[:2]
        nw, nh = rotated_size(W, H, angle)
        nw, nh = int(round(nw)), int(round(nh))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle, 1.0)   # cv2: positive = counter-clockwise
        M[0, 2] += nw / 2 - W / 2
        M[1, 2] += nh / 2 - H / 2
        img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, (nw, nh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        invalid = valid < 250
    if crop is not None:
        H, W = img.shape[:2]
        x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(crop, (W, H, W, H)))
        x0, x1 = max(0, min(x0, W - 1)), max(1, min(x1, W))
        y0, y1 = max(0, min(y0, H - 1)), max(1, min(y1, H))
        img = img[y0:y1, x0:x1]
        if invalid is not None:
            invalid = invalid[y0:y1, x0:x1]
    if invalid is not None and not invalid.any():
        invalid = None
    return np.ascontiguousarray(img), invalid
