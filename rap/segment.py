# --------------------------------------------
# SAM 3 segmentation
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""SAM 3 wrapper: clicks/boxes (SAM 1-style instance prompts) and text prompts."""
import numpy as np
import torch
from PIL import Image


class Segmenter:
    def __init__(self, device='cuda', confidence=0.5):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.device = device
        self.model = build_sam3_image_model(device=device, enable_inst_interactivity=True)
        self.proc = Sam3Processor(self.model, device=device, confidence_threshold=confidence)

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16)

    @torch.no_grad()
    def set_image(self, img: np.ndarray):
        with self._autocast():
            return self.proc.set_image(Image.fromarray(img))

    @torch.no_grad()
    def from_prompts(self, state, points=(), box=None, prev_logits=None):
        """Mask from positive/negative clicks and/or a box, in image pixels.

        ``points`` is a list of (x, y, label) with label 1 = include, 0 = exclude.
        Returns (mask bool HxW, score, low-res logits for the next refinement).
        """
        kw = {}
        if len(points):
            pts = np.asarray(points, dtype=np.float32)
            kw['point_coords'], kw['point_labels'] = pts[:, :2], pts[:, 2].astype(np.int32)
        if box is not None:
            kw['box'] = np.asarray(box, dtype=np.float32)[None]
        # A single click is ambiguous (part vs. whole), so let SAM propose three
        # masks and keep the best; later clicks refine the previous mask.
        multi = prev_logits is None and len(points) == 1 and box is None
        if prev_logits is not None:
            kw['mask_input'] = prev_logits[None]
        with self._autocast():
            masks, scores, logits = self.model.predict_inst(state, multimask_output=multi, **kw)
        i = int(np.argmax(scores))
        return masks[i] > 0, float(scores[i]), logits[i]

    @torch.no_grad()
    def from_text(self, state, prompt):
        """All instances matching a noun phrase. Returns [(mask, score)], best first."""
        with self._autocast():
            self.proc.reset_all_prompts(state)
            out = self.proc.set_text_prompt(prompt=prompt, state=state)
        masks = out['masks'][:, 0].cpu().numpy()
        scores = out['scores'].float().cpu().numpy()
        order = np.argsort(-scores)
        return [(masks[i], float(scores[i])) for i in order]
