# --------------------------------------------
# DDColor colorization restorer
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""DDColor (Kang et al., ICCV 2023) automatic colorization, Apache-2.0.

The network predicts the ab (chroma) channels of CIE Lab at 512x512 from the
lightness; they are upsampled and recombined with the full-resolution L of
the crop, so detail and tone are untouched. The control scales the chroma,
which makes every candidate a cheap colour-space conversion.
"""
import json
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import Param, Restorer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'DDColor'))
REPO = 'piddnad/ddcolor_modelscope'


def _import_ddcolor():
    # DDColor ships its own `basicsr`, which clashes with CodeFormer's copy. Load
    # its model under a temporary `basicsr` and put the previous one back.
    saved = {k: v for k, v in sys.modules.items() if k == 'basicsr' or k.startswith('basicsr.')}
    for k in saved:
        del sys.modules[k]
    try:
        for name in ('basicsr', 'basicsr.archs'):
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(_ROOT, *name.split('.'))]
            sys.modules[name] = mod
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from ddcolor.model import DDColor
        return DDColor
    finally:
        for k in [k for k in sys.modules if k == 'basicsr' or k.startswith('basicsr.')]:
            del sys.modules[k]
        sys.modules.update(saved)


class DDColorRestorer(Restorer):
    key = 'ddcolor'
    label = 'DDColor · Colorize'
    group = 'color'
    description = ('Automatic colorization of black-and-white or faded photos. Keeps the original '
                   'lightness and detail and only adds colour.')
    params = [Param('chroma', 'Color intensity', 0, 1.5, 1.0, 0.05,
                    sweep=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                    hint='0 = black and white, 1 = the model\'s colours, above 1 = more saturated.')]
    size = 512

    def __init__(self, weights_dir, device):
        from huggingface_hub import hf_hub_download
        DDColor = _import_ddcolor()
        cfg = json.load(open(hf_hub_download(REPO, 'config.json')))
        self.net = DDColor(**cfg)
        sd = torch.load(hf_hub_download(REPO, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
        self.net.load_state_dict(sd, strict=True)
        self.net.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def prepare(self, img, mask=None):
        H, W = img.shape[:2]
        rgb = img.astype(np.float32) / 255
        L = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., :1]
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        l_small = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[..., :1]
        gray = cv2.cvtColor(np.concatenate([l_small, np.zeros_like(l_small), np.zeros_like(l_small)], -1),
                            cv2.COLOR_LAB2RGB)
        x = torch.from_numpy(gray.transpose(2, 0, 1))[None].to(self.device)
        ab = self.net(x).float()
        ab = F.interpolate(ab, size=(H, W), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        return {'L': L, 'ab': ab}

    def render(self, state, values):
        lab = np.concatenate([state['L'], state['ab'] * float(values.get('chroma', 1.0))], -1)
        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        return (rgb.clip(0, 1) * 255).round().astype(np.uint8)
