# --------------------------------------------
# CodeFormer face restorer
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""CodeFormer (Zhou et al., NeurIPS 2022) blind face restoration.

Faces are detected in the crop, aligned to 512x512, restored and pasted back
with CodeFormer's parsing-based blending. Everything that does not depend on
the fidelity weight (detection, alignment, the inverse-warp masks) is done
once in ``prepare``; ``render_many`` then restores every (face, fidelity)
pair in one batched pass, which matters on a shared GPU where each kernel
launch has to wait for a time slice.

CodeFormer is released under the S-Lab License 1.0 (non-commercial).
"""
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import Param, Restorer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'CodeFormer'))
# Face-parsing classes kept in the blend mask (CodeFormer's MASK_COLORMAP).
_KEEP = torch.tensor([0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0],
                     dtype=torch.float32)


def _import_codeformer():
    # CodeFormer vendors basicsr, whose package __init__ pulls in the whole
    # training stack (and a version.py that only its setup.py generates).
    # Register bare packages so only the modules we use get imported.
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    for name in ('basicsr', 'basicsr.archs'):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(_ROOT, *name.split('.'))]
            sys.modules[name] = mod
    from basicsr.archs.codeformer_arch import CodeFormer as Net
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import adain_npy, bgr2gray
    return Net, FaceRestoreHelper, bgr2gray, adain_npy


class _PerSampleWeight:
    """Lets CodeFormer's scalar ``w`` differ per batch element.

    The network does ``if w > 0: x = x + fuse(..., w)`` and scales by ``w``
    inside; w = 0 in a batch element then gives exactly the unfused output.
    """

    def __init__(self, t):
        self.t = t

    def __gt__(self, other):
        return True

    def __mul__(self, x):
        return self.t * x

    __rmul__ = __mul__


def _gaussian_blur(x, ksize, sigma):
    """cv2.GaussianBlur (BORDER_REFLECT_101) for a (B, 1, H, W) tensor."""
    r = ksize // 2
    g = torch.exp(-(torch.arange(ksize, device=x.device, dtype=x.dtype) - r) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), g.view(1, 1, -1, 1))


class CodeFormer(Restorer):
    key = 'codeformer'
    group = 'face'
    label = 'CodeFormer · Faces'
    description = ('Blind face restoration. Detects faces inside the region and restores '
                   'each one. Low fidelity gives cleaner faces, high fidelity stays closer '
                   'to the input identity.')
    params = [Param('w', 'Fidelity', 0, 1, 0.5, 0.05,
                    sweep=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    hint='0 = best-looking face, 1 = most faithful to the input.')]
    batch = 16
    weight_files = ('codeformer/codeformer.pth', 'codeformer/detection_Resnet50_Final.pth',
                    'codeformer/parsing_parsenet.pth')

    def __init__(self, weights_dir, device):
        Net, Helper, self._bgr2gray, self._adain = _import_codeformer()
        self.device = device
        self.net = Net(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                       connect_list=['32', '64', '128', '256']).to(device)
        ckpt = torch.load(f'{weights_dir}/codeformer/codeformer.pth', map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['params_ema'])
        self.net.eval().requires_grad_(False)
        # Loads detection/parsing weights from third_party/CodeFormer/weights/facelib,
        # which is symlinked to weights/codeformer.
        self.helper = Helper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
                             save_ext='png', use_parse=True, device=device)
        self.keep = _KEEP.to(device)
        from ..models import gpu_gb
        if gpu_gb() < 20:   # smaller face batches from the start on small GPUs
            self.batch = 4 if gpu_gb() >= 12 else 2

    def extra_modules(self):
        return [self.helper.face_detector, self.helper.face_parse]

    @torch.no_grad()
    def prepare(self, img, mask=None):
        h = self.helper
        h.clean_all()
        h.read_image(np.ascontiguousarray(img[:, :, ::-1]))   # upsamples crops shorter than 512 px
        h.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        h.align_warp_face()
        base = h.input_img
        Hu, Wu = base.shape[:2]
        faces = []
        for crop, affine in zip(h.cropped_faces, h.affine_matrices):
            inv = cv2.invertAffineTransform(affine)
            # Square-mask feathering from paste_faces_to_input_image; depends only on the warp.
            mask = cv2.warpAffine(np.ones((512, 512), np.float32), inv, (Wu, Hu))
            erosion = cv2.erode(mask, np.ones((2, 2), np.uint8))
            w_edge = int(erosion.sum() ** 0.5) // 20
            center = cv2.erode(erosion, np.ones((w_edge * 2, w_edge * 2), np.uint8))
            soft = cv2.GaussianBlur(center, (w_edge * 2 + 1, w_edge * 2 + 1), 0)
            t = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)
            faces.append({'input': ((t - 0.5) / 0.5).to(self.device), 'crop': crop, 'inv': inv,
                          'erosion': erosion[..., None], 'soft': soft[..., None]})
        return {'base': base.astype(np.float32), 'size': img.shape[:2], 'faces': faces, 'gray': h.is_gray}

    @torch.no_grad()
    def shrink(self):
        """Smaller face batches after running out of GPU memory; False once at one face."""
        if self.batch <= 1:
            return False
        self.batch //= 2
        return True

    def render_many(self, state, values_list):
        faces = state['faces']
        H, W = state['size']
        if not faces:
            img = state['base']
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return [np.ascontiguousarray(img[:, :, ::-1]).round().astype(np.uint8) for _ in values_list]

        ws = [float(v.get('w', 0.5)) for v in values_list]
        pairs = [(i, j) for i in range(len(ws)) for j in range(len(faces))]
        restored, parse = [], []
        for k in range(0, len(pairs), self.batch):
            chunk = pairs[k:k + self.batch]
            x = torch.stack([faces[j]['input'] for _, j in chunk])
            w = torch.tensor([ws[i] for i, _ in chunk], device=self.device).view(-1, 1, 1, 1)
            out = self.net(x, w=_PerSampleWeight(w), adain=True)[0].clamp(-1, 1)
            rgb = ((out + 1) * 127.5).round()                                   # uint8 values, float
            # Blend mask from the parsing net on the restored face, as in paste-back.
            labels = self.helper.face_parse((rgb / 255 - 0.5) / 0.5)[0].argmax(1)
            m = self.keep[labels][:, None]
            m = _gaussian_blur(_gaussian_blur(m, 101, 11), 101, 11)
            m[..., :10, :] = 0
            m[..., -10:, :] = 0
            m[..., :, :10] = 0
            m[..., :, -10:] = 0
            restored.append(rgb.byte().permute(0, 2, 3, 1).cpu().numpy())
            parse.append((m[:, 0] / 255).cpu().numpy())
        restored = np.concatenate(restored)
        parse = np.concatenate(parse)

        base = state['base']
        Hu, Wu = base.shape[:2]
        outs = []
        for i in range(len(ws)):
            img = base.copy()
            for j, f in enumerate(faces):
                k = i * len(faces) + j
                face = np.ascontiguousarray(restored[k][:, :, ::-1])
                if state['gray']:
                    face = self._adain(self._bgr2gray(face), f['crop'])
                pasted = f['erosion'] * cv2.warpAffine(face, f['inv'], (Wu, Hu))
                pm = cv2.warpAffine(parse[k], f['inv'], (Wu, Hu), flags=3)[..., None]
                soft = np.where(pm < f['soft'], pm, f['soft'])
                img = soft * pasted + (1 - soft) * img
            img = img.clip(0, 255).astype(np.uint8)
            if (Hu, Wu) != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            outs.append(np.ascontiguousarray(img[:, :, ::-1]))
        return outs

    def render(self, state, values):
        return self.render_many(state, [values])[0]

    def info(self, state):
        n = len(state['faces'])
        return f'{n} face{"s" if n != 1 else ""} detected' if n else 'No face detected in this region'
