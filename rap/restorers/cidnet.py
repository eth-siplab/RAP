# --------------------------------------------
# HVI-CIDNet low-light restorer
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""HVI-CIDNet (Yan et al., CVPR 2025) low-light enhancement, MIT.

Uses the author-recommended "Generalization" weights. Its brightness and
saturation controls only enter the final HVI -> RGB transform, so the network
runs once per crop and every setting is a cheap transform of the same output.
"""
import os
import sys

import torch
import torch.nn.functional as F

from .base import Param, Restorer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'HVI-CIDNet'))
REPO = 'Fediory/HVI-CIDNet-Generalization'


class CIDNetLowLight(Restorer):
    key = 'cidnet'
    label = 'HVI-CIDNet · Low light'
    group = 'light'
    description = ('Brightens under-exposed photos and suppresses the noise and colour casts that come '
                   'with them.')
    params = [Param('bright', 'Brightness', 0.3, 2.0, 1.0, 0.05,
                    sweep=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0],
                    hint='Output brightness relative to the model\'s choice.'),
              Param('sat', 'Saturation', 0, 2, 0.85, 0.05, hint='Colour saturation of the result.')]

    def __init__(self, weights_dir, device):
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from net.CIDNet import CIDNet
        self.net = CIDNet.from_pretrained(REPO).to(device).eval().requires_grad_(False)
        self.net.trans.gated = True
        self.net.trans.gated2 = True
        self.device = device

    @torch.no_grad()
    def prepare(self, img, mask=None):
        H, W = img.shape[:2]
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = F.pad(x, (0, -W % 8, 0, -H % 8), mode='reflect')
        trans, cap = self.net.trans, {}
        phvit = trans.PHVIT
        trans.PHVIT = lambda hvi: cap.setdefault('hvi', hvi)   # stop before the final transform
        try:
            self.net(x)
        finally:
            trans.PHVIT = phvit
        return {'hvi': cap['hvi'], 'size': (H, W)}

    @torch.no_grad()
    def render(self, state, values):
        t = self.net.trans
        t.alpha, t.alpha_s = float(values.get('bright', 1.0)), float(values.get('sat', 0.85))
        H, W = state['size']
        y = t.PHVIT(state['hvi']).clamp(0, 1)[0, :, :H, :W]
        return (y.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
