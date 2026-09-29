# --------------------------------------------
# Editing sessions and regions
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Per-image editing state: regions, their masks, restorer choice and cached work.

Coordinates exchanged with the browser are in *preview* pixels (the image
downscaled so its long side fits ``PREVIEW_MAX``); everything stored here and
all restoration run at full resolution. Only what is sent back is downscaled.
"""
import io
import itertools
import uuid

import cv2
import numpy as np
from PIL import Image

PREVIEW_MAX = 2560
BACKGROUND = 'bg'

_ids = itertools.count(1)


def reserve_region_ids(n):
    """Make new region ids start above n (after sessions were loaded from disk)."""
    global _ids
    _ids = itertools.count(max(next(_ids), n + 1))


def encode_png(img, level=1):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG', compress_level=level)
    return buf.getvalue()


def stroke_mask(shape, pts, radius):
    """Round-capped polyline of the given radius, as a bool mask."""
    m = np.zeros(shape, np.uint8)
    r = max(1, int(round(radius)))
    pts = np.round(np.asarray(pts, np.float64)).astype(np.int32)
    for p in pts:
        cv2.circle(m, (int(p[0]), int(p[1])), r, 1, -1)
    if len(pts) > 1:
        cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 1, thickness=2 * r)
    return m > 0


def feather(mask, px):
    """Soft alpha in [0, 1] from a binary mask; px is the blur sigma."""
    a = mask.astype(np.float32)
    if px > 0:
        k = int(px * 3) * 2 + 1
        a = cv2.GaussianBlur(a, (k, k), px)
    return a


class Region:
    def __init__(self, name, mask=None, source='click'):
        self.id = BACKGROUND if mask is None else f'r{next(_ids)}'
        self.name = name
        self.mask = mask                  # full-res bool, None for the background
        self.soft = None                  # optional full-res uint8 alpha (matting); never edited in place
        self.source = source              # 'click' | 'text' | 'background'
        self.points = []                  # SAM clicks, full-res (x, y, label)
        self.box = None
        self.logits = None
        self.score = None
        self.feather = 2.0                # in full-res pixels
        self.restorer = 'none'
        self.values = {}
        self.adjust = {}                  # see rap/adjust.py; empty = identity
        self.cache = {}                   # (restorer, crop box) -> prepared state
        self.renders = {}                 # (restorer, crop box, values) -> blob url
        self.visible = True
        self.strokes = []                 # brush edits (full-res), re-applied after SAM updates

    # state saved in the session history (everything except caches)
    FIELDS = ('name', 'source', 'points', 'box', 'logits', 'score', 'feather', 'restorer', 'values',
              'adjust', 'visible', 'strokes', 'soft')

    def state(self):
        st = {k: getattr(self, k) for k in self.FIELDS}
        st['values'], st['adjust'] = dict(self.values), dict(self.adjust)
        st['points'], st['strokes'] = list(self.points), list(self.strokes)
        st['mask'] = None if self.mask is None else (np.packbits(self.mask), self.mask.shape)
        return st

    def load_state(self, st):
        for k in self.FIELDS:
            setattr(self, k, st[k])
        if st['mask'] is None:
            self.mask = None
        else:
            bits, shape = st['mask']
            self.mask = np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)

    def apply_strokes(self, mask):
        for pts, radius, add in self.strokes:
            m = stroke_mask(mask.shape, pts, radius)
            mask = (mask | m) if add else (mask & ~m)
        return mask

    def crop_box(self, W, H, context=0.15):
        """Mask bounding box plus a context margin, as (x0, y0, x1, y1)."""
        if self.mask is None:
            return (0, 0, W, H)
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        m = int(max(32, context * max(x1 - x0, y1 - y0)))
        return (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))

    def alpha(self, box):
        x0, y0, x1, y1 = box
        if self.mask is None:
            return np.ones((y1 - y0, x1 - x0), np.float32)
        if self.soft is not None:
            return self.soft[y0:y1, x0:x1].astype(np.float32) / 255
        return feather(self.mask[y0:y1, x0:x1], self.feather)


class Session:
    def __init__(self, img, seg_state):
        self.id = uuid.uuid4().hex[:12]
        self.img = img
        self.H, self.W = img.shape[:2]
        self.scale = min(1.0, PREVIEW_MAX / max(self.H, self.W))   # full -> preview
        self.seg_state = seg_state
        bg = Region('Background', source='background')
        self.regions = {bg.id: bg}
        self.order = [bg.id]              # bottom to top
        self.blobs = {}                   # token -> (bytes, mime)
        self.exif = None                  # original EXIF bytes (orientation reset), kept on export
        self.history, self.future = [], []   # undo / redo stacks of snapshots

    def rebase(self, img, seg_state):
        """Make img the new base image (after apply / crop); regions start over."""
        self._reset(img, seg_state)

    # ---- history -----------------------------------------------------
    def snapshot(self):
        return {'img': self.img, 'order': list(self.order),
                'regions': {rid: r.state() for rid, r in self.regions.items()}}

    def push(self):
        """Record the current state before a change (clears redo)."""
        self.history.append(self.snapshot())
        del self.history[:-60]
        self.future.clear()

    def _restore(self, snap, embed):
        if snap['img'] is not self.img:
            self.img = snap['img']
            self.H, self.W = self.img.shape[:2]
            self.scale = min(1.0, PREVIEW_MAX / max(self.H, self.W))
            self.seg_state = embed(self.img)
            for r in self.regions.values():   # caches refer to the other image
                r.cache, r.renders = {}, {}
        old = self.regions
        self.regions = {}
        for rid, st in snap['regions'].items():
            r = old.get(rid) or Region(st['name'], mask=np.zeros(1, bool) if st['mask'] is not None else None)
            r.id = rid
            r.load_state(st)
            self.regions[rid] = r
        self.order = list(snap['order'])

    def undo(self, embed):
        if not self.history:
            return False
        self.future.append(self.snapshot())
        self._restore(self.history.pop(), embed)
        return True

    def redo(self, embed):
        if not self.future:
            return False
        self.history.append(self.snapshot())
        self._restore(self.future.pop(), embed)
        return True

    def _reset(self, img, seg_state):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.scale = min(1.0, PREVIEW_MAX / max(self.H, self.W))
        self.seg_state = seg_state
        bg = Region('Background', source='background')
        self.regions = {bg.id: bg}
        self.order = [bg.id]
        self.blobs = {}

    # ---- coordinates -------------------------------------------------
    def to_full(self, v):
        return v / self.scale

    def preview_size(self):
        return (max(1, round(self.W * self.scale)), max(1, round(self.H * self.scale)))

    def preview(self, img):
        """Downscale a full-res crop/image for the browser."""
        if self.scale >= 1:
            return img
        h, w = img.shape[:2]
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def box_to_preview(self, box):
        return [round(v * self.scale) for v in box]

    # ---- blobs served to the browser ---------------------------------
    def put_blob(self, data, mime='image/png'):
        token = uuid.uuid4().hex[:16]
        self.blobs[token] = (data, mime)
        return f'/api/blob/{self.id}/{token}'

    def drop_blobs(self, urls):
        for u in urls:
            self.blobs.pop(u.rsplit('/', 1)[-1], None)

    # ---- regions -----------------------------------------------------
    def add_region(self, region):
        self.regions[region.id] = region
        self.order.append(region.id)
        return region

    def remove_region(self, rid):
        if rid != BACKGROUND and rid in self.regions:
            del self.regions[rid]
            self.order.remove(rid)

    def mask_png(self, region):
        """RGBA crop of the feathered mask (alpha channel) at preview scale."""
        box = region.crop_box(self.W, self.H)
        a = (region.alpha(box) * 255).round().astype(np.uint8)
        rgba = np.zeros(a.shape + (4,), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = a
        return self.preview(rgba), box

    def outline_png(self, region):
        """Mask as a flat RGBA overlay (full image, preview scale) for display."""
        if region.mask is None:
            return None
        m = self.preview(region.mask.astype(np.uint8) * 255)
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = m
        return rgba

    def composite(self, renders, transparent=False, base=None):
        """renders: {region_id: (full-res crop, crop box, alpha or None)}; bottom to top.

        alpha None means the region's feathered mask. `base` is the photo the layers
        go on (default: the image itself). With transparent=True the base is empty and
        the result is RGBA.
        """
        if not transparent:
            out = (self.img if base is None else base).astype(np.float32)
            for rid in self.order:
                if rid not in renders:
                    continue
                crop, box, alpha = renders[rid]
                x0, y0, x1, y1 = box
                a = (self.regions[rid].alpha(box) if alpha is None else alpha)[..., None]
                out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + crop.astype(np.float32) * a
            return out.round().clip(0, 255).astype(np.uint8)
        pre = np.zeros((self.H, self.W, 3), np.float32)     # premultiplied colour
        acc = np.zeros((self.H, self.W, 1), np.float32)
        for rid in self.order:
            if rid not in renders:
                continue
            crop, box, alpha = renders[rid]
            x0, y0, x1, y1 = box
            a = (self.regions[rid].alpha(box) if alpha is None else alpha)[..., None]
            pre[y0:y1, x0:x1] = pre[y0:y1, x0:x1] * (1 - a) + crop.astype(np.float32) * a
            acc[y0:y1, x0:x1] = acc[y0:y1, x0:x1] * (1 - a) + a
        rgb = np.where(acc > 1e-4, pre / np.maximum(acc, 1e-4), 0)
        return np.concatenate([rgb, acc * 255], -1).round().clip(0, 255).astype(np.uint8)
