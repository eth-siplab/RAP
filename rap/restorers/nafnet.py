# --------------------------------------------
# NAFNet motion deblurring
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""NAFNet (Chen et al., ECCV 2022) motion deblurring, GoPro weights, MIT.

The control is how many times the network is applied: one pass is the
published setting, extra passes help with stronger blur. Large crops are
processed in overlapping tiles.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .base import Param, Restorer
from .nafnet_arch import NAFNetLocal


def _tiled(fn, x, tile=768, overlap=64):
    _, _, H, W = x.shape
    if H * W <= tile * tile * 2:
        return fn(x)
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x[:, :1])
    ramp = torch.ones(tile, device=x.device)
    ramp[:overlap] = torch.linspace(0.1, 1, overlap, device=x.device)
    ramp[-overlap:] = torch.linspace(1, 0.1, overlap, device=x.device)
    win = ramp[:, None] * ramp[None, :]
    step = tile - overlap
    for y in list(range(0, max(H - tile, 0), step)) + [max(H - tile, 0)]:
        for x0 in list(range(0, max(W - tile, 0), step)) + [max(W - tile, 0)]:
            patch = x[:, :, y:y + tile, x0:x0 + tile]
            h, w = patch.shape[-2:]
            out[:, :, y:y + h, x0:x0 + w] += fn(patch) * win[:h, :w]
            weight[:, :, y:y + h, x0:x0 + w] += win[:h, :w]
    return out / weight


class NAFNetDeblur(Restorer):
    key = 'nafnet_deblur'
    label = 'NAFNet · Motion blur'
    group = 'restore'
    description = ('Removes camera-shake and motion blur (trained on GoPro). Use more passes for '
                   'stronger blur.')
    weight_files = ('nafnet/NAFNet-GoPro-width64.pth',)
    params = [Param('passes', 'Passes', 1, 3, 1, 1, sweep=[1, 2, 3],
                    hint='How many times to apply the deblurring network.')]

    def __init__(self, weights_dir, device):
        self.device = device
        net = NAFNetLocal(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        sd = torch.load(f'{weights_dir}/nafnet/NAFNet-GoPro-width64.pth', map_location='cpu', weights_only=True)
        net.load_state_dict(sd['params'], strict=True)
        self.net = net.to(device).eval().requires_grad_(False)

    def prepare(self, img, mask=None):
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        return {'passes': [x]}

    @torch.no_grad()
    def render_many(self, state, values_list):
        need = max(int(v.get('passes', 1)) for v in values_list)
        while len(state['passes']) <= need:   # cache each pass for the other settings
            state['passes'].append(_tiled(lambda t: self.net(t).clamp(0, 1), state['passes'][-1]))
        return [(state['passes'][int(v.get('passes', 1))][0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                for v in values_list]

    def render(self, state, values):
        return self.render_many(state, [values])[0]
