# --------------------------------------------
# FLUX.2 [klein] generative fill
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Text-guided generative fill with FLUX.2 [klein] 4B (Apache-2.0, distilled to
4 steps). Replaces what is inside a region with what the prompt describes; on
the background layer it regenerates everything outside the other regions
(i.e. replaces the background). Loaded on first use (16 GB of weights). On a
GPU with less than 28 GB the weights stay in CPU memory and stream in layer by
layer: four variants in 10-20 s with a peak of about 6.5 GB (measured; streaming
whole components instead was 3-4x slower). The variants are generated in one
batch, or one at a time if that does not fit.
"""
import cv2
import numpy as np
import torch
from PIL import Image

from .base import Param, Restorer

REPO = 'black-forest-labs/FLUX.2-klein-4B'


class GenFill(Restorer):
    key = 'genfill'
    label = 'FLUX.2 · Generative fill'
    group = 'generate'
    description = ('Replaces the region with what you describe, e.g. "a red vintage bicycle" or, on the '
                   'background, "a beach at sunset". Each variant is a different generation.')
    params = [Param('variant', 'Variant', 1, 4, 1, 1, sweep=[1, 2, 3, 4], hint='Different generations; pick the best.'),
              Param('strength', 'Repaint strength', 0.3, 1.0, 1.0, 0.05,
                    hint='Lower keeps more of the original structure and colours (good for "make it red").')]
    prompt_hint = 'Describe what should be there…'
    needs_mask = True
    allow_bg = True          # on the background: mask = everything outside the other regions
    uses_foreground = True
    context = 0.4
    blend = 'box'
    edits_photo = True
    size = 1024
    self_offloading = False   # True when the pipeline keeps its components in CPU memory between uses

    def __init__(self, weights_dir, device):
        self.device, self.pipe = device, None
        self.one_by_one = False   # set after running out of memory with a batch

    def defaults(self):
        return {**{p.key: p.default for p in self.params}, 'prompt': ''}

    def warm(self):
        self._load()

    def _load(self):
        if self.pipe is None:
            from diffusers import Flux2KleinInpaintPipeline
            from ..models import gpu_gb
            pipe = Flux2KleinInpaintPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
            if gpu_gb() >= 28:
                pipe.to(self.device)
            else:
                pipe.enable_sequential_cpu_offload(device=self.device)
                self.self_offloading = True
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
        return self.pipe

    def prepare(self, img, mask=None):
        H, W = img.shape[:2]
        s = self.size / max(H, W)
        w, h = max(16, int(round(W * s / 16)) * 16), max(16, int(round(H * s / 16)) * 16)
        grow = max(3, int(0.012 * max(H, W)))
        hole = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
        soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), grow / 2)[..., None]
        return {'img': img, 'soft': np.maximum(soft, mask[..., None].astype(np.float32)),
                'small': Image.fromarray(img).resize((w, h), Image.LANCZOS),
                'small_mask': Image.fromarray(hole * 255).resize((w, h), Image.NEAREST)}

    @torch.no_grad()
    def render_many(self, state, values_list):
        out = [None] * len(values_list)
        groups = {}
        for i, v in enumerate(values_list):
            groups.setdefault((str(v.get('prompt', '')), float(v.get('strength', 1.0))), []).append(i)
        img, soft = state['img'], state['soft']
        H, W = img.shape[:2]
        for (prompt, strength), idx in groups.items():
            seeds = sorted({int(values_list[i].get('variant', 1)) for i in idx})
            images = self._generate(prompt, strength, seeds, state['small'], state['small_mask'])
            by_seed = {}
            for s, g in zip(seeds, images):
                g = np.asarray(g.resize((W, H), Image.LANCZOS), np.float32)
                by_seed[s] = (img * (1 - soft) + g * soft).round().clip(0, 255).astype(np.uint8)
            for i in idx:
                out[i] = by_seed[int(values_list[i].get('variant', 1))]
        return out

    def _generate(self, prompt, strength, seeds, im, mk):
        pipe = self._load()
        kw = dict(prompt=prompt, image=im, mask_image=mk, height=im.height, width=im.width, strength=strength,
                  num_inference_steps=4, guidance_scale=1.0)
        gen = lambda s: torch.Generator(self.device).manual_seed(1000 + s)
        if not self.one_by_one and len(seeds) > 1:
            try:
                return pipe(**kw, num_images_per_prompt=len(seeds), generator=[gen(s) for s in seeds]).images
            except torch.cuda.OutOfMemoryError:
                self.one_by_one = True
            self._reset_offload()
        try:
            return [pipe(**kw, generator=[gen(s)]).images[0] for s in seeds]
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:   # outside the except block, so the failed attempt's memory can go
            self._reset_offload()
            raise torch.cuda.OutOfMemoryError('FLUX.2 generative fill does not fit in GPU memory')

    def _reset_offload(self):
        """After a failed call the offloaded component that was running stays on the GPU; send it back."""
        if self.self_offloading:
            self.pipe.maybe_free_model_hooks()
        torch.cuda.empty_cache()

    def render(self, state, values):
        return self.render_many(state, [values])[0]
