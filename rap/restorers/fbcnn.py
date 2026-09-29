# --------------------------------------------
# FBCNN JPEG artifact restorer
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os

import numpy as np
import torch

from .base import Param, Restorer
from .fbcnn_arch import FBCNN


class _FBCNNBase(Restorer):
    weight = ''

    @classmethod
    def available(cls, weights_dir):
        return os.path.isfile(os.path.join(weights_dir, 'fbcnn', cls.weight))

    def __init__(self, weights_dir, device):
        self.device = device
        self.net = FBCNN()
        sd = torch.load(f'{weights_dir}/fbcnn/{self.weight}', map_location='cpu', weights_only=True)
        self.net.load_state_dict(sd, strict=True)
        self.net.eval().requires_grad_(False).to(device)

    def to_cond(self, value):
        raise NotImplementedError

    def from_code(self, code):
        raise NotImplementedError

    big = 2_500_000     # above this many pixels, work in tiles instead of caching features
    tile = 1024
    overlap = 64

    @torch.no_grad()
    def prepare(self, img, mask=None):
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        H, W = img.shape[:2]
        if H * W <= self.big:
            feats, code, size = self.net.encode(x)
            return {'feats': feats, 'code': code, 'size': size}
        # Large crop: keep only the input; estimate the degradation on a
        # full-resolution centre tile (downscaling would erase JPEG blocking).
        t = self.tile
        y0, x0 = max(0, (H - t) // 2) // 8 * 8, max(0, (W - t) // 2) // 8 * 8
        _, code, _ = self.net.encode(x[:, :, y0:y0 + t, x0:x0 + t])
        return {'x': x, 'code': code}

    def _decode_batch(self, feats, conds, size):
        # Condition-batch size bounded by pixels so memory stays ~4 GB.
        n = max(1, min(4, int(4e6 / (size[0] * size[1]))))
        out = []
        for i in range(0, len(conds), n):
            out.append(self.net.decode(feats, conds[i:i + n], size).clamp(0, 1))
        return torch.cat(out)

    @torch.no_grad()
    def render_many(self, state, values_list):
        # The condition only enters the decoder, and the batch-1 encoder
        # features broadcast against a batch of conditions.
        key = self.params[0].key
        conds = torch.from_numpy(np.stack([np.asarray(self.to_cond(v[key]), np.float32).reshape(-1)
                                           for v in values_list])).to(self.device)
        if 'feats' in state:
            y = self._decode_batch(state['feats'], conds, state['size'])
        else:
            y = self._tiled(state['x'], conds)
        return list((y.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())

    def _tiled(self, x, conds):
        _, _, H, W = x.shape
        t, ov = self.tile, self.overlap
        out = torch.zeros((len(conds), 3, H, W), device=x.device)
        weight = torch.zeros((1, 1, H, W), device=x.device)

        def spans(n):
            # Starts are multiples of 8 so the JPEG 8x8 grid stays aligned; the
            # last span runs to the border (up to 7 px longer than a tile).
            if n <= t:
                return [(0, n)]
            starts = list(range(0, n - t, (t - ov) // 8 * 8)) + [(n - t) // 8 * 8]
            starts = sorted(set(starts))
            return [(a, b) for a, b in zip(starts, [s + t for s in starts[:-1]] + [n])]

        def ramp(n, lo, hi):
            r = torch.ones(n, device=x.device)
            if lo:
                r[:ov] = torch.linspace(0.05, 1, ov, device=x.device)
            if hi:
                r[-ov:] = torch.linspace(1, 0.05, ov, device=x.device)
            return r
        ys, xs = spans(H), spans(W)
        for i, (y0, y1) in enumerate(ys):
            for j, (x0, x1) in enumerate(xs):
                feats, _, size = self.net.encode(x[:, :, y0:y1, x0:x1])
                win = ramp(y1 - y0, i > 0, i < len(ys) - 1)[:, None] * ramp(x1 - x0, j > 0, j < len(xs) - 1)[None, :]
                out[:, :, y0:y1, x0:x1] += self._decode_batch(feats, conds, size) * win
                weight[:, :, y0:y1, x0:x1] += win
        return out / weight

    def render(self, state, values):
        return self.render_many(state, [values])[0]

    def estimate(self, state):
        p = self.params[0]
        code = state['code'][0].float().cpu().numpy()
        if not 0.02 < code[0] < 0.98:
            return {}  # estimate pinned at the end of its range: out of the model's depth
        v = self.from_code(code)
        return {p.key: float(np.clip(round(v / p.step) * p.step, p.min, p.max))}


class FBCNNDeblock(_FBCNNBase):
    key = 'fbcnn_deblock'
    label = 'FBCNN · JPEG artifacts'
    description = 'Removes JPEG blocking and ringing. Lower QF means stronger smoothing.'
    weight = 'fbcnn_deblock.pth'
    params = [Param('qf', 'Quality factor', 1, 100, 50, 1,
                    sweep=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                    hint='Treat the region as if it were saved at this JPEG quality.')]

    def to_cond(self, qf):
        return [1 - qf / 100]

    def from_code(self, code):
        return (1 - float(code[0])) * 100
