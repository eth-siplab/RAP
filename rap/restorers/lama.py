# --------------------------------------------
# LaMa erase
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""LaMa (Suvorov et al., WACV 2022) inpainting, used to erase a region.

Uses the TorchScript export of big-lama distributed with IOPaint (Apache-2.0).
The control is how far the region mask is grown before erasing, which is what
usually decides whether edges, halos and soft shadows disappear with it; all
expansion values are inpainted in one batch.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import Param, Restorer


def _grow(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


class LaMa(Restorer):
    key = 'lama'
    label = 'LaMa · Erase'
    group = 'remove'
    description = ('Fast inpainting: erases the region and fills it from its surroundings. Best for '
                   'scratches, text, watermarks, wires and small objects.')
    params = [Param('expand', 'Mask expansion', 0, 40, 8, 1,
                    sweep=[0, 4, 8, 12, 16, 24, 32, 40],
                    hint='Grow the erased area by this many pixels to take edges, halos and soft '
                         'shadows with it.')]
    needs_mask = True
    context = 0.3
    blend = 'box'
    edits_photo = True
    weight_files = ('lama/big-lama.pt',)
    max_side = 2048   # beyond this LaMa runs downscaled and only the hole is upsampled
    batch = 8

    def __init__(self, weights_dir, device):
        self.device = device
        self.net = torch.jit.load(f'{weights_dir}/lama/big-lama.pt', map_location=device).eval()
        from ..models import gpu_gb
        if gpu_gb() < 12:
            self.max_side = 1536

    def prepare(self, img, mask=None):
        return {'img': img, 'mask': mask}

    @torch.no_grad()
    def render_many(self, state, values_list):
        img, mask = state['img'], state['mask']
        H, W = img.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        h, w = max(8, round(H * s)), max(8, round(W * s))
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        xs = F.interpolate(x, size=(h, w), mode='area') if s < 1 else x
        pad = (0, -w % 8, 0, -h % 8)
        holes = [_grow(mask, int(v.get('expand', 8))) for v in values_list]
        outs = []
        for i in range(0, len(holes), self.batch):
            chunk = holes[i:i + self.batch]
            m = torch.from_numpy(np.stack(chunk)).to(self.device)[:, None].float()
            ms = (F.interpolate(m, size=(h, w), mode='area') > 0).float() if s < 1 else m
            # TorchScript would re-optimise for every new crop size; plain execution is faster here.
            with torch.jit.optimized_execution(False):
                pred = self.net(F.pad(xs.expand(len(chunk), -1, -1, -1), pad, mode='reflect'),
                                F.pad(ms, pad))[:, :, :h, :w]
            if s < 1:
                pred = F.interpolate(pred, size=(H, W), mode='bicubic', align_corners=False)
            # Soft edge so the fill blends into the untouched pixels.
            soft = torch.from_numpy(np.stack([cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.5) for c in chunk]))
            soft = torch.maximum(soft.to(self.device)[:, None], m)
            y = x * (1 - soft) + pred.clamp(0, 1) * soft
            outs.extend((y.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())
        return outs

    def shrink(self):
        """Run at a lower resolution after running out of GPU memory; False once at the minimum."""
        if self.max_side <= 768:
            return False
        self.max_side = max(768, int(self.max_side * 0.7))
        return True

    def render(self, state, values):
        return self.render_many(state, [values])[0]
