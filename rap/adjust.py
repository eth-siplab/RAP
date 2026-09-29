# --------------------------------------------
# Per-region tonal and colour adjustments
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Per-region tonal adjustments, applied after restoration.

The browser implements the same pipeline (rap/static/app.js, applyAdjust)
for live preview; this module is the reference used for export. Keep the two
in sync. All blur radii are fractions of the image's long side, so a
downscaled preview and the full-resolution export look the same.

1. mix      m = floor(orig + strength * (restored - orig) + 0.5)          (8 bit)
2. tone     per-channel 256-entry LUT: linearise, white balance (temperature
            on R/B, tint on G) and exposure; back to sRGB; levels (blacks,
            whites); shadows/highlights curves; contrast around 0.5
3. colour   saturation and vibrance around Rec.709 luma
4. local    clarity (mid-tone local contrast on luma) and dehaze (dark-channel
            haze estimate, atmospheric light = white)
5. detail   unsharp mask, [1 2 1]/4 blur (Gaussian at other scales)
6. vignette in full-image coordinates
7. clip to [0, 1], round to 8 bit
"""
import cv2
import numpy as np

SPEC = [
    {'key': 'exposure', 'label': 'Exposure', 'group': 'Light', 'min': -2, 'max': 2, 'step': 0.05, 'default': 0, 'unit': ' EV'},
    {'key': 'contrast', 'label': 'Contrast', 'group': 'Light', 'min': -0.9, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'highlights', 'label': 'Highlights', 'group': 'Light', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'shadows', 'label': 'Shadows', 'group': 'Light', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'whites', 'label': 'Whites', 'group': 'Light', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'blacks', 'label': 'Blacks', 'group': 'Light', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'temperature', 'label': 'Temperature', 'group': 'Color', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'tint', 'label': 'Tint', 'group': 'Color', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'vibrance', 'label': 'Vibrance', 'group': 'Color', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'saturation', 'label': 'Saturation', 'group': 'Color', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'clarity', 'label': 'Clarity', 'group': 'Effects', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'dehaze', 'label': 'Dehaze', 'group': 'Effects', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
    {'key': 'sharpen', 'label': 'Sharpen', 'group': 'Effects', 'min': 0, 'max': 3, 'step': 0.05, 'default': 0, 'unit': ''},
    {'key': 'vignette', 'label': 'Vignette', 'group': 'Effects', 'min': -1, 'max': 1, 'step': 0.01, 'default': 0, 'unit': ''},
]
DEFAULTS = {s['key']: s['default'] for s in SPEC}
WB_STOPS = 0.3        # temperature/tint ±1 = ±0.3 stops
CURVE_NORM = 9.4815   # 1 / max of x(1-x)^3, so the shadow/highlight bumps peak at 1
CLARITY_R = 0.01      # clarity radius, fraction of the long side
DEHAZE_R = 0.03


def is_identity(adj, has_restorer):
    if has_restorer and adj.get('strength', 1.0) != 1.0:
        return False
    return all(adj.get(k, v) == v for k, v in DEFAULTS.items())


def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(y):
    y = np.maximum(y, 0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


def tone_luts(adj):
    g = lambda k: float(adj.get(k, 0.0))
    ev, c, t, ti = g('exposure'), g('contrast'), g('temperature'), g('tint')
    hl, sh, wh, bl = g('highlights'), g('shadows'), g('whites'), g('blacks')
    lo, hi = -0.1 * bl, 1 - 0.1 * wh
    lin = _srgb_to_lin(np.arange(256) / 255.0)
    luts = []
    for gain in (2 ** (WB_STOPS * t), 2 ** (-WB_STOPS * ti), 2 ** (-WB_STOPS * t)):
        s = _lin_to_srgb(lin * (2 ** ev) * gain)
        s = (s - lo) / (hi - lo)
        sc = np.clip(s, 0, 1)
        s = s + 0.2 * CURVE_NORM * (sh * sc * (1 - sc) ** 3 + hl * sc ** 3 * (1 - sc))
        luts.append((s - 0.5) * (1 + c) + 0.5)
    return np.stack(luts, 1).astype(np.float32)   # (256, 3)


def _box3(x, r):
    """Three box blurs of radius r (approx. Gaussian), replicated borders."""
    if r < 1:
        return x
    for _ in range(3):
        x = cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)
    return x


def _luma(v):
    return v @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def apply(orig, restored, adj, scale=1.0, offset=(0, 0), full_size=None):
    """orig, restored: uint8 (H, W, 3) crops.

    scale: preview/full ratio (keeps sharpen's apparent radius), offset/full_size:
    where the crop sits in the full image (x0, y0), (W, H), for the vignette and
    the size-relative blur radii.
    """
    if restored is None:
        m = orig
    else:
        s = float(adj.get('strength', 1.0))
        m = np.floor(orig.astype(np.float32) + s * (restored.astype(np.float32) - orig) + 0.5)
        m = m.clip(0, 255).astype(np.uint8)
    H, W = m.shape[:2]
    FW, FH = full_size or (W, H)
    long_side = max(FW, FH)
    luts = tone_luts(adj)
    v = np.stack([luts[m[..., i], i] for i in range(3)], -1)
    g = lambda k: float(adj.get(k, 0.0))

    sat, vib = g('saturation'), g('vibrance')
    if sat or vib:
        L = _luma(v)[..., None]
        spread = v.max(-1, keepdims=True) - v.min(-1, keepdims=True)
        v = L + (1 + sat) * (1 + vib * (1 - np.clip(spread, 0, 1))) * (v - L)

    cl = g('clarity')
    if cl:
        L = _luma(v)
        base = _box3(L, int(round(CLARITY_R * long_side)))
        w = 1 - (2 * np.clip(L, 0, 1) - 1) ** 2
        v = v + (cl * (L - base) * w)[..., None]

    dz = g('dehaze')
    if dz > 0:
        haze = _box3(np.clip(v.min(-1), 0, 1), int(round(DEHAZE_R * long_side)))
        t = np.maximum(1 - 0.6 * dz * haze, 0.25)[..., None]
        v = 1 - (1 - v) / t
    elif dz < 0:
        k = -0.5 * dz
        v = v * (1 - k) + 0.85 * k

    amt = g('sharpen')
    if amt:
        if scale >= 1:
            k = np.array([0.25, 0.5, 0.25], np.float32)
            blur = cv2.sepFilter2D(v, -1, k, k, borderType=cv2.BORDER_REPLICATE)
        else:  # keep the same apparent radius as in the downscaled preview
            blur = cv2.GaussianBlur(v, (0, 0), 0.7 / scale, borderType=cv2.BORDER_REPLICATE)
        v = v + amt * (v - blur)

    vg = g('vignette')
    if vg:
        x0, y0 = offset
        ys = (np.arange(H, dtype=np.float32) + y0 + 0.5) / FH - 0.5
        xs = (np.arange(W, dtype=np.float32) + x0 + 0.5) / FW - 0.5
        d = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / np.sqrt(0.5)
        e = np.clip((d - 0.35) / 0.65, 0, 1)
        v = v * (1 - 0.9 * vg * e * e * (3 - 2 * e))[..., None]

    return np.floor(v.clip(0, 1) * 255 + 0.5).astype(np.uint8)
