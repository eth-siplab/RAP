# --------------------------------------------
# PiSA-SR real-world restoration and upscaling
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""PiSA-SR (Sun et al., CVPR 2025) as a 1x restorer.

PiSA-SR adds two LoRAs to SD-2.1-base: a pixel-level one (l2 loss, removes
degradations) and a semantic-level one (LPIPS + CSD, adds detail). Inference
is a single UNet step, and the prediction is linear in the two strengths:

    pred = l_pix * pred_pix + l_sem * (pred_sem - pred_pix)

where pred_pix comes from base+pix weights and pred_sem from base+pix+sem.
We therefore keep the two merged UNets, run each once per crop in
``prepare`` and leave only a VAE decode for ``render``.

The upstream code pins diffusers 0.25 / peft 0.9; here the LoRA deltas are
merged by hand into a stock diffusers UNet instead. Large latents are tiled
with Gaussian weights (variance 0.01) and colours are matched with AdaIN, as in
the upstream inference code (https://github.com/csslc/PiSA-SR, Apache-2.0).
"""
import copy
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .base import Param, Restorer

# stabilityai/stable-diffusion-2-1-base was taken down; this community mirror
# has byte-identical weights (same LFS hashes as other mirrors).
SD21_BASE = 'sd2-community/stable-diffusion-2-1-base'
LORA_ALPHA = 8  # peft LoraConfig default, which PiSA-SR relies on


def _lora_delta(A, B, base_shape, scaling):
    if len(base_shape) == 2:  # Linear
        return (B @ A) * scaling
    if base_shape[2:] == (1, 1):  # 1x1 conv
        return (B[:, :, 0, 0] @ A[:, :, 0, 0])[:, :, None, None] * scaling
    # kxk conv: A is (r, in, k, k), B is (out, r, 1, 1)
    return F.conv2d(A.permute(1, 0, 2, 3), B).permute(1, 0, 2, 3) * scaling


def _merge_lora(unet, sd, tags):
    """Add the LoRA deltas whose adapter name ends with one of ``tags``."""
    params = dict(unet.named_parameters())
    lora = sd['state_dict_unet']
    n = 0
    for k in lora:
        if '.lora_A.' not in k:
            continue
        module, rest = k.split('.lora_A.')
        adapter = rest.split('.')[0]
        if not adapter.endswith(tags):
            continue
        rank = sd['lora_rank_unet_' + adapter.rsplit('_', 1)[1]]
        A = lora[k].float()
        B = lora[f'{module}.lora_B.{adapter}.weight'].float()
        W = params[module + '.weight']
        W.data += _lora_delta(A, B, tuple(W.shape), LORA_ALPHA / rank).to(W.device)
        n += 1
    return n


def _gaussian_window(size, device):
    x = torch.arange(size, device=device, dtype=torch.float32)
    mid = (size - 1) / 2
    g = torch.exp(-(x - mid) ** 2 / (2 * size ** 2 * 0.01))
    return g[:, None] * g[None, :]


def _tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = tile - overlap
    n = math.ceil((length - tile) / stride) + 1
    return [min(i * stride, length - tile) for i in range(n)]


def _adain(x, ref):
    """Match per-channel mean/std of x to ref (both B,C,H,W in [0,1])."""
    xm, xs = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-5
    rm, rs = ref.mean((2, 3), keepdim=True), ref.std((2, 3), keepdim=True) + 1e-5
    return (x - xm) / xs * rs + rm


class PiSASR(Restorer):
    key = 'pisasr'
    label = 'PiSA-SR · Real-world'
    description = ('One-step diffusion restoration for real photos with mixed degradations. '
                   'Detail controls how much texture is generated; Fidelity controls how hard '
                   'degradations are removed.')
    params = [
        Param('sem', 'Detail', 0, 1.5, 1.0, 0.05,
              sweep=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
              hint='0 = no generated texture (cleanest, may look smooth); higher = more '
                   'generated detail (sharper, may hallucinate).'),
        Param('pix', 'Fidelity', 0, 1.5, 1.0, 0.05,
              hint='Strength of the pixel-level correction (denoising / artifact removal).'),
    ]

    weight_files = ('pisasr/pisa_sr.pkl',)
    tile = 96          # latent tiles (768 px)
    tile_overlap = 32
    tile_batch = 8     # latent tiles per UNet call
    decode_pixels = 4e6   # pixels per VAE-decode batch
    min_side = 512     # the model was trained at 512; smaller crops are upsampled first

    def __init__(self, weights_dir, device, dtype=torch.float16):
        self.device, self.dtype = device, dtype
        sd = torch.load(f'{weights_dir}/pisasr/pisa_sr.pkl', map_location='cpu', weights_only=False)

        # LoRAs are merged in float32 on the CPU; only the two float16 UNets go to the GPU.
        unet = UNet2DConditionModel.from_pretrained(SD21_BASE, subfolder='unet', torch_dtype=torch.float32)
        n_pix = _merge_lora(unet, sd, ('_pix',))
        unet_sem = copy.deepcopy(unet)
        n_sem = _merge_lora(unet_sem, sd, ('_sem',))
        assert n_pix == n_sem == 258, (n_pix, n_sem)
        self.unet_pix = unet.to(device, dtype).eval().requires_grad_(False)
        self.unet_sem = unet_sem.to(device, dtype).eval().requires_grad_(False)
        del unet, unet_sem, sd

        self.vae = AutoencoderKL.from_pretrained(SD21_BASE, subfolder='vae', torch_dtype=dtype)
        self.vae.to(device).eval().requires_grad_(False)
        self.vae.enable_tiling()
        self.scale = self.vae.config.scaling_factor
        from ..models import gpu_gb
        gb = gpu_gb()
        if gb >= 20:
            self._vae_tile(1536)
        else:   # smaller pieces from the start on small GPUs, instead of failing and retrying
            self.tile_batch, self.decode_pixels = (2, 1.5e6) if gb >= 12 else (1, 1e6)
            self._vae_tile(1024 if gb >= 12 else 768)

        tok = CLIPTokenizer.from_pretrained(SD21_BASE, subfolder='tokenizer')
        te = CLIPTextModel.from_pretrained(SD21_BASE, subfolder='text_encoder').to(device, dtype)
        with torch.no_grad():
            ids = tok([''], max_length=tok.model_max_length, padding='max_length', return_tensors='pt').input_ids
            self.prompt = te(ids.to(device))[0].to(dtype)
        del te
        self.t = torch.tensor([1], device=device).long()

    @torch.no_grad()
    def _unets(self, lat):
        """Both UNet predictions, tiled over the latent if it is large."""
        _, _, h, w = lat.shape
        if h * w <= self.tile * self.tile:
            p = self.prompt.expand(1, -1, -1)
            return self.unet_pix(lat, self.t, p).sample, self.unet_sem(lat, self.t, p).sample
        tile = min(self.tile, h, w)
        win = _gaussian_window(tile, lat.device)
        coords = [(y, x) for y in _tile_starts(h, tile, self.tile_overlap)
                  for x in _tile_starts(w, tile, self.tile_overlap)]
        out_pix = torch.zeros_like(lat, dtype=torch.float32)
        out_sem = torch.zeros_like(out_pix)
        weight = torch.zeros((1, 1, h, w), device=lat.device)
        for i in range(0, len(coords), self.tile_batch):
            batch = coords[i:i + self.tile_batch]
            tiles = torch.cat([lat[:, :, y:y + tile, x:x + tile] for y, x in batch])
            p = self.prompt.expand(len(batch), -1, -1)
            pp = self.unet_pix(tiles, self.t, p).sample.float()
            ps = self.unet_sem(tiles, self.t, p).sample.float()
            for j, (y, x) in enumerate(batch):
                out_pix[:, :, y:y + tile, x:x + tile] += pp[j] * win
                out_sem[:, :, y:y + tile, x:x + tile] += ps[j] * win
                weight[:, :, y:y + tile, x:x + tile] += win
        return (out_pix / weight).to(lat.dtype), (out_sem / weight).to(lat.dtype)

    @torch.no_grad()
    def prepare(self, img, mask=None):
        h0, w0 = img.shape[:2]
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        s = max(1.0, self.min_side / min(h0, w0))
        if s > 1:
            x = F.interpolate(x, scale_factor=s, mode='bicubic', align_corners=False).clamp(0, 1)
        h, w = x.shape[-2:]
        xp = F.pad(x, (0, -w % 8, 0, -h % 8), mode='reflect')
        lat = self.vae.encode((xp * 2 - 1).to(self.dtype)).latent_dist.mode() * self.scale
        pred_pix, pred_sem = self._unets(lat)
        return {'lat': lat, 'pix': pred_pix, 'sem': pred_sem, 'src': x, 'size0': (h0, w0), 'size': (h, w)}

    batch = 4

    @torch.no_grad()
    def render_many(self, state, values_list):
        lats = torch.cat([state['lat'] - (v.get('pix', 1.0) * state['pix']
                                          + v.get('sem', 1.0) * (state['sem'] - state['pix']))
                          for v in values_list])
        h, w = state['size']
        n = max(1, min(self.batch, int(self.decode_pixels / (h * w))))   # keep VAE-decode memory bounded
        outs = []
        for i in range(0, len(lats), n):
            out = self.vae.decode(lats[i:i + n] / self.scale).sample
            out = (out[:, :, :h, :w].float().clamp(-1, 1) + 1) / 2
            out = _adain(out, state['src']).clamp(0, 1)
            if state['size0'] != (h, w):
                out = F.interpolate(out, size=state['size0'], mode='area')
            outs.extend((out.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())
        return outs

    def render(self, state, values):
        return self.render_many(state, [values])[0]

    def _vae_tile(self, px):
        self.vae.tile_sample_min_size = px
        self.vae.tile_latent_min_size = px // 8

    def shrink(self):
        """Smaller UNet batches, VAE batches and VAE tiles after running out of GPU memory;
        False at the minimum."""
        if self.tile_batch > 1:
            self.tile_batch //= 2
            return True
        if self.decode_pixels > 5e5:
            self.decode_pixels /= 2
            return True
        smaller = [px for px in (1024, 768, 512) if px < self.vae.tile_sample_min_size]
        if smaller:
            self._vae_tile(smaller[0])
            return True
        return False

    @torch.no_grad()
    def upscale(self, img, scale, detail=0.8):
        """Super-resolve a whole image by ``scale`` (what PiSA-SR was trained for).

        The input is bicubic-upsampled to the target size first, as in training.
        """
        H, W = img.shape[:2]
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        up = F.interpolate(x, size=(round(H * scale), round(W * scale)), mode='bicubic', align_corners=False)
        up = (up.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        state = self.prepare(up)
        return self.render(state, {'sem': detail, 'pix': 1.0})
