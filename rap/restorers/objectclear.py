# --------------------------------------------
# ObjectClear object and shadow removal
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""ObjectClear (Zhao et al., CVPR 2026): removes an object together with its
shadow and reflection, given only the object mask. Built on SDXL-Inpainting.

It works at short side 512. Instead of upscaling its whole output (which would
blur untouched areas), only the area it changed goes back into the full-res
crop: that area is the object mask grown a little, plus the region its
cross-attention marks as the object's effect (same rule as its own
attention-guided fusion).

Weights are under the NTU S-Lab License 1.0 (non-commercial), like CodeFormer.
The pipeline is loaded on first use (about 8 GB of GPU memory).
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from .base import Param, Restorer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'ObjectClear'))
SEED = 42
PROMPT = 'remove the instance of object'   # ObjectClear's fixed instruction


class ObjectClear(Restorer):
    key = 'objectclear'
    label = 'ObjectClear · Remove + shadow'
    group = 'remove'
    description = ('Removes an object together with its shadow and reflection, and fills in the '
                   'background. Slower (diffusion); each variant is a different random fill.')
    params = [Param('variant', 'Variant', 1, 4, 1, 1, sweep=[1, 2, 3, 4],
                    hint='Different random fills of the removed area; pick the cleanest.'),
              Param('strength', 'Removal strength', 1, 5, 2.5, 0.25,
                    hint='Higher removes more aggressively (shadows, reflections); lower keeps more '
                         'of the original background.')]
    needs_mask = True
    context = 0.6      # shadows often reach well beyond the object
    blend = 'box'
    edits_photo = True
    size = 512         # short side the model works at
    steps = 20

    def __init__(self, weights_dir, device):
        self.device = device
        self.pipe = None

    def warm(self):
        self._load()

    def _load(self):
        if self.pipe is None:
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            from objectclear.pipelines import ObjectClearPipeline
            self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
                'jixin0101/ObjectClear', torch_dtype=torch.float16, apply_attention_guided_fusion=True,
                variant='fp16')
            self.pipe.set_progress_bar_config(disable=True)
            # The prompt never changes: encode it once, then drop both text encoders (1.6 GB)
            # before the rest goes to the GPU, so the peak stays low on small GPUs.
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            with torch.no_grad():
                pe, npe, ppe, nppe = self.pipe.encode_prompt(PROMPT, device=self.device, num_images_per_prompt=1,
                                                             do_classifier_free_guidance=True)
            self.embeds = {'prompt_embeds': pe, 'negative_prompt_embeds': npe,
                           'pooled_prompt_embeds': ppe, 'negative_pooled_prompt_embeds': nppe}
            self.pipe.text_encoder = self.pipe.text_encoder_2 = None
            torch.cuda.empty_cache()
            self.pipe.to(self.device)
        return self.pipe

    def prepare(self, img, mask=None):
        H, W = img.shape[:2]
        s = self.size / min(H, W)
        # its attention-map resizing needs dimensions divisible by 64
        w, h = max(64, round(W * s / 64) * 64), max(64, round(H * s / 64) * 64)
        return {'img': img, 'mask': mask,
                'small': Image.fromarray(img).resize((w, h), Image.BICUBIC),
                'small_mask': Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)}

    def _run(self, state, strength, seeds):
        pipe = self._load()
        im, mk = state['small'], state['small_mask']
        n = len(seeds)
        # Its built-in fusion pairs every output with an original image but only
        # decodes one original per call; hand it one per output.
        pp = pipe.image_processor.postprocess

        def postprocess(image, output_type='pil', **kw):
            out = pp(image, output_type=output_type, **kw)
            return out * n if output_type == 'pil' and isinstance(out, list) and len(out) == 1 else out

        pipe.image_processor.postprocess = postprocess
        try:
            res = pipe(**self.embeds, image=im, mask_image=mk,
                       generator=[torch.Generator(self.device).manual_seed(SEED + k) for k in seeds],
                       num_images_per_prompt=n, num_inference_steps=self.steps, guidance_scale=strength,
                       height=im.height, width=im.width, return_attn_map=True)
        finally:
            pipe.image_processor.postprocess = pp
        return res.images, res.attns

    def _paste(self, state, gen, attn):
        """Put the changed area of a 512-scale result into the full-res crop."""
        img, mask = state['img'], state['mask']
        H, W = img.shape[:2]
        w, h = gen.size
        # Effect area from the attention map, as in ObjectClear's attention-guided fusion.
        a = (np.asarray(attn.resize((w, h), Image.NEAREST), np.float32) > 128).astype(np.float32)
        a = np.maximum(a, cv2.GaussianBlur(
            cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))), (9, 9), 2))
        obj = np.asarray(state['small_mask'], np.float32) / 255
        obj = cv2.GaussianBlur(cv2.dilate(obj, np.ones((7, 7), np.uint8)), (0, 0), 2)
        a = cv2.resize(np.maximum(a, obj), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        g = cv2.resize(np.asarray(gen, np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return (img * (1 - a) + g * a).round().clip(0, 255).astype(np.uint8)

    @torch.no_grad()
    def render_many(self, state, values_list):
        out = [None] * len(values_list)
        groups = {}
        for i, v in enumerate(values_list):
            groups.setdefault(float(v.get('strength', 2.5)), []).append(i)
        for strength, idx in groups.items():
            seeds = [int(values_list[i].get('variant', 1)) - 1 for i in idx]
            uniq = sorted(set(seeds))
            gens, attns = self._run(state, strength, uniq)
            by_seed = {s: self._paste(state, g, a) for s, g, a in zip(uniq, gens, attns)}
            for i, s in zip(idx, seeds):
                out[i] = by_seed[s]
        return out

    def render(self, state, values):
        return self.render_many(state, [values])[0]
