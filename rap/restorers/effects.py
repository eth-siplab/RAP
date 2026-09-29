# --------------------------------------------
# Effects: transparent, solid colour, blur, pixelate, lens blur
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Background and privacy effects: transparent, solid colour, blur, pixelate
and depth-aware lens blur. None of these are learned restorers, but they share
the same region / slider / sweep machinery."""
import colorsys

import cv2
import numpy as np

from .base import Param, Restorer


class Transparent(Restorer):
    key = 'transparent'
    label = 'Transparent (cut out)'
    group = 'effect'
    description = 'Removes the background: exports a PNG/WebP with transparency around the selected regions.'
    params = []
    bg_only = True
    blend = 'transparent'

    def prepare(self, img, mask=None):
        return img

    def render(self, state, values):
        return state


class SolidColor(Restorer):
    key = 'color_fill'
    label = 'Solid color'
    group = 'effect'
    description = 'Replaces the background with a flat colour.'
    params = [Param('lightness', 'Lightness', 0, 1, 1.0, 0.01, sweep=[0, 0.2, 0.5, 0.8, 0.95, 1.0],
                    hint='0 = black, 1 = white (at zero saturation).'),
              Param('hue', 'Hue', 0, 360, 210, 1, hint='Colour of the fill.'),
              Param('saturation', 'Saturation', 0, 1, 0.0, 0.01, hint='0 = gray scale.')]
    bg_only = True

    def prepare(self, img, mask=None):
        return img.shape

    def render(self, state, values):
        h, l, s = values.get('hue', 210) / 360, values.get('lightness', 1.0), values.get('saturation', 0.0)
        rgb = np.array(colorsys.hls_to_rgb(h, l, s)) * 255
        return np.broadcast_to(rgb.round().astype(np.uint8), state).copy()


def _part_size(mask):
    """Typical size (px) of the separate parts of a mask, e.g. of each face in a
    region holding several faces; the effect strength scales with it."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= max(16, 0.02 * areas.max())] if len(areas) else areas
    return float(np.sqrt(np.median(areas))) if len(areas) else float(np.sqrt(mask.size))


class Blur(Restorer):
    key = 'blur'
    label = 'Blur (privacy)'
    group = 'effect'
    description = 'Gaussian blur, e.g. to hide faces, number plates or screens.'
    params = [Param('amount', 'Amount', 0, 1, 0.5, 0.05, sweep=[0.1, 0.25, 0.5, 0.75, 1.0],
                    hint='Blur radius relative to the size of each selected part (e.g. each face).')]
    needs_mask = True

    def prepare(self, img, mask=None):
        return {'img': img, 'size': _part_size(mask)}

    def render(self, state, values):
        sigma = max(0.5, float(values.get('amount', 0.5)) * 0.25 * state['size'])
        return cv2.GaussianBlur(state['img'], (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


class Pixelate(Restorer):
    key = 'pixelate'
    label = 'Pixelate (privacy)'
    group = 'effect'
    description = 'Mosaic, e.g. to hide faces, number plates or screens.'
    params = [Param('block', 'Block size', 0, 1, 0.5, 0.05, sweep=[0.1, 0.25, 0.5, 0.75, 1.0],
                    hint='Mosaic cell size relative to the size of each selected part (e.g. each face).')]
    needs_mask = True

    def prepare(self, img, mask=None):
        return {'img': img, 'size': _part_size(mask)}

    def render(self, state, values):
        img = state['img']
        H, W = img.shape[:2]
        cell = max(2, int(round(2 + float(values.get('block', 0.5)) * 0.3 * state['size'])))
        small = cv2.resize(img, (max(1, W // cell), max(1, H // cell)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


class LensBlur(Restorer):
    """Depth-aware background blur. The regions above stay sharp (they are drawn
    on top); the blur is normalised by a weight that excludes them so their
    colours do not bleed into the background around their edges."""
    key = 'lens_blur'
    label = 'Lens blur (depth)'
    group = 'effect'
    description = ('Blurs the background like a wide-aperture lens, more the farther it is from the '
                   'selected subject (depth from Depth Anything V2).')
    params = [Param('amount', 'Blur', 0, 1, 0.5, 0.05, sweep=[0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0],
                    hint='Maximum blur radius (up to 2.5% of the image size).'),
              Param('falloff', 'Depth falloff', 0, 1, 0.5, 0.05,
                    hint='How quickly the blur grows with distance from the subject\'s depth.')]
    bg_only = True
    uses_foreground = True
    levels = 6

    def __init__(self, depth_model):
        self.depth = depth_model

    def prepare(self, img, mask=None):
        d = self.depth(img)
        fg = mask if mask is not None and mask.any() else None
        focus = float(np.median(d[fg])) if fg is not None else float(np.percentile(d, 97))
        w = np.ones(img.shape[:2], np.float32) if fg is None else (1 - cv2.GaussianBlur(
            cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2))
        return {'img': img.astype(np.float32), 'depth': d, 'focus': focus, 'w': np.clip(w, 0, 1)}

    def _blur(self, x, sigma):
        # large sigmas on a downscaled copy: same look, a fraction of the cost
        f = max(1, int(sigma // 6))
        H, W = x.shape[:2]
        small = cv2.resize(x, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA) if f > 1 else x
        small = cv2.GaussianBlur(small, (0, 0), max(0.3, sigma / f), borderType=cv2.BORDER_REFLECT)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) if f > 1 else small

    def render(self, state, values):
        img, d, w = state['img'], state['depth'], state['w']
        H, W = img.shape[:2]
        amount, falloff = float(values.get('amount', 0.5)), float(values.get('falloff', 0.5))
        rmax = amount * 0.025 * max(H, W)
        if rmax < 0.5:
            return img.round().astype(np.uint8)
        # blur radius per pixel from depth distance to the subject
        scale = 0.05 + 0.6 * (1 - falloff)
        r = rmax * np.clip(np.abs(d - state['focus']) / scale, 0, 1)
        # smooth the radius map so depth edges do not turn into hard steps
        r = cv2.GaussianBlur(r.astype(np.float32), (0, 0), max(1.0, 0.004 * max(H, W)))
        radii = [0] + [rmax * (i / (self.levels - 1)) ** 1.5 for i in range(1, self.levels)]
        stack = [img]
        for rad in radii[1:]:
            num = self._blur(img * w[..., None], rad / 2)
            den = self._blur(w, rad / 2)[..., None]
            stack.append(np.where(den > 1e-3, num / np.maximum(den, 1e-3), img))
        # interpolate between the two nearest levels
        out = np.zeros_like(img)
        for i in range(self.levels - 1):
            lo, hi = radii[i], radii[i + 1]
            t = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
            sel = ((r >= lo) & ((r < hi) | (i == self.levels - 2)))[..., None]
            out = np.where(sel, stack[i] * (1 - t) + stack[i + 1] * t, out)
        return out.round().clip(0, 255).astype(np.uint8)
