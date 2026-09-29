# --------------------------------------------
# Subject matting and depth
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Subject matting (BiRefNet, MIT) and monocular depth (Depth Anything V2 Small,
Apache-2.0), both loaded on first use."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BIREFNET = ('ZhengPeng7/BiRefNet', 'e2bf8e4460')    # repo, pinned revision (remote code)
DEPTH = 'depth-anything/Depth-Anything-V2-Small-hf'


class SubjectModel:
    size = 1024

    def __init__(self, device):
        self.device, self.net = device, None

    def warm(self):
        if self.net is None:
            from transformers import AutoModelForImageSegmentation
            self.net = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET[0], trust_remote_code=True, revision=BIREFNET[1]).to(self.device).eval().half()

    @torch.no_grad()
    def __call__(self, img):
        """Soft foreground alpha (uint8, same size as img)."""
        self.warm()
        H, W = img.shape[:2]
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        a = self.net(((x - mean) / std).half())[-1].sigmoid().float()
        a = F.interpolate(a, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        return (a.clamp(0, 1) * 255).round().byte().cpu().numpy()


class DepthModel:
    def __init__(self, device):
        self.device, self.pipe = device, None

    def warm(self):
        if self.pipe is None:
            from transformers import pipeline
            self.pipe = pipeline('depth-estimation', model=DEPTH, device=self.device)

    def extra_modules(self):
        return [self.pipe.model] if self.pipe is not None else []

    @torch.no_grad()
    def __call__(self, img):
        """Relative inverse depth in [0, 1] (1 = nearest), same size as img."""
        self.warm()
        d = self.pipe(Image.fromarray(img))['predicted_depth']
        d = F.interpolate(d[None, None].float(), size=img.shape[:2], mode='bilinear', align_corners=False)[0, 0]
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return d.cpu().numpy()
