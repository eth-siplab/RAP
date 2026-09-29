# --------------------------------------------
# SCUNet and DRUNet denoisers
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Kai Zhang's denoisers from KAIR: SCUNet (blind, real noise) and DRUNet (noise level as a control).

SCUNet runs both official real-noise models once per crop, the PSNR one and
the GAN one, and the control blends their outputs. DRUNet takes the noise
level as an extra input plane; its starting value is estimated from the crop.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import Param, Restorer
from .kair_arch import SCUNet, UNetRes
from .nafnet import _tiled


def _load(net, path):
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return net.eval().requires_grad_(False)


def _to_tensor(img, device):
    return torch.from_numpy(img).to(device).permute(2, 0, 1)[None].float() / 255


def _to_uint8(y):
    return (y.clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def _padded(net, x, m):
    """Run ``net`` on ``x`` mirror-padded to a multiple of ``m``. Repeating the edge pixels instead (what
    the reference code does) turns noise into streaks the networks leave untouched near the border."""
    H, W = x.shape[-2:]
    ph, pw = -H % m, -W % m
    if not ph and not pw:
        return net(x)
    mode = 'reflect' if ph < H and pw < W else 'replicate'
    return net(F.pad(x, (0, pw, 0, ph), mode=mode))[..., :H, :W]


def noise_sigma(img):
    """Immerkaer's fast noise estimate, per colour channel, in 0-255 levels."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    f = img.astype(np.float32)
    per = [np.abs(cv2.filter2D(f[..., c], -1, k))[1:-1, 1:-1].mean() for c in range(3)]
    return float(np.sqrt(np.pi / 2) * np.mean(per) / 6)


class SCUNetReal(Restorer):
    key = 'scunet'
    label = 'SCUNet · Real noise'
    group = 'restore'
    description = ('Blind denoising for real photos: phones, high ISO, dark shots, grainy scans. Slide towards '
                   'Texture for the GAN model, which keeps crisper detail.')
    weight_files = ('kair/scunet_color_real_psnr.pth', 'kair/scunet_color_real_gan.pth')
    params = [Param('texture', 'Texture', 0, 1, 0, 0.05, sweep=[0, 0.25, 0.5, 0.75, 1],
                    hint='0: the smooth, faithful model · 1: the GAN model, crisper but may invent fine detail.')]
    big = 4_000_000     # above this many pixels, run in tiles

    def __init__(self, weights_dir, device):
        self.device = device
        self.psnr = _load(SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64),
                          f'{weights_dir}/kair/scunet_color_real_psnr.pth').to(device)
        self.gan = _load(SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64),
                         f'{weights_dir}/kair/scunet_color_real_gan.pth').to(device)

    def _run(self, net, x):
        fn = lambda t: _padded(net, t, 64)
        if x.shape[2] * x.shape[3] <= self.big:
            return fn(x)
        return _tiled(fn, x, tile=1024, overlap=128)

    @torch.no_grad()
    def prepare(self, img, mask=None):
        x = _to_tensor(img, self.device)
        return {'psnr': self._run(self.psnr, x)[0].clamp(0, 1).half(),
                'gan': self._run(self.gan, x)[0].clamp(0, 1).half()}

    @torch.no_grad()
    def render_many(self, state, values_list):
        out = []
        for v in values_list:
            t = float(v.get('texture', 0))
            out.append(_to_uint8(state['psnr'].float() * (1 - t) + state['gan'].float() * t))
        return out

    def render(self, state, values):
        return self.render_many(state, [values])[0]


class DRUNetDenoise(Restorer):
    key = 'drunet'
    label = 'DRUNet · Noise level'
    group = 'restore'
    description = ('Removes noise at the strength you choose. The noise level starts at an estimate for this '
                   'region; lower keeps more detail, higher smooths more.')
    weight_files = ('kair/drunet_color.pth',)
    params = [Param('sigma', 'Noise level σ', 0, 75, 15, 1, sweep=[5, 10, 15, 20, 25, 35, 50, 75],
                    hint='Standard deviation of the noise to remove (0–255 scale).')]
    big = 2_000_000

    def __init__(self, weights_dir, device):
        self.device = device
        self.net = _load(UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                                 downsample_mode='strideconv', upsample_mode='convtranspose', bias=False),
                         f'{weights_dir}/kair/drunet_color.pth').to(device)

    def _denoise(self, x, sigmas):
        """x: (1, 3, H, W); returns (len(sigmas), 3, H, W)."""
        s = torch.tensor(sigmas, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1) / 255
        inp = torch.cat([x.expand(len(sigmas), -1, -1, -1), s.expand(-1, 1, *x.shape[-2:])], 1)
        return _padded(self.net, inp, 8)

    @torch.no_grad()
    def prepare(self, img, mask=None):
        return {'x': _to_tensor(img, self.device), 'img': img, 'out': {}}

    @torch.no_grad()
    def render_many(self, state, values_list):
        x, cache = state['x'], state['out']
        todo = sorted({int(round(v.get('sigma', 15))) for v in values_list} - cache.keys())
        HW = x.shape[2] * x.shape[3]
        if HW > self.big:
            for sg in todo:
                cache[sg] = _tiled(lambda t, sg=sg: self._denoise(t, [sg]), x, tile=768, overlap=64)[0].half()
        else:
            n = max(1, min(4, int(2e6 / HW)))   # batch bounded by pixels
            for i in range(0, len(todo), n):
                for sg, y in zip(todo[i:i + n], self._denoise(x, todo[i:i + n])):
                    cache[sg] = y.clamp(0, 1).half()
        return [_to_uint8(cache[int(round(v.get('sigma', 15)))].float()) for v in values_list]

    def render(self, state, values):
        return self.render_many(state, [values])[0]

    def estimate(self, state):
        img = state['img']
        H, W = img.shape[:2]
        c = img[max(0, H // 2 - 512):H // 2 + 512, max(0, W // 2 - 512):W // 2 + 512]
        return {'sigma': float(np.clip(round(noise_sigma(c)), 1, 75))}
