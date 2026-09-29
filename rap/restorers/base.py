# --------------------------------------------
# Restorer interface
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Param:
    key: str
    label: str
    min: float
    max: float
    default: float
    step: float
    # Values precomputed when a region is assigned this restorer, so dragging
    # the slider can show a result without a server round trip.
    sweep: list = field(default_factory=list)
    hint: str = ''


class Restorer:
    """A restoration model with continuous controls.

    Work is split in two so that sweeping a control is cheap:
    ``prepare`` runs everything that does not depend on the controls (once per
    crop) and ``render`` turns that state plus control values into an image.
    Images are RGB uint8 arrays of shape (H, W, 3).
    """

    key = ''
    label = ''
    description = ''
    params: list = []
    group = 'restore'   # UI section: restore | face | remove | color | light
    needs_mask = False  # prepare() also gets the region mask (bool, same crop)
    context = 0.15      # crop margin around the region, as a fraction of its size
    blend = 'mask'      # 'mask': paste with the feathered region mask;
                        # 'box': the output already blends itself, replace the crop;
                        # 'transparent': (background) leave it empty
    bg_only = False     # only offered for the background layer
    uses_foreground = False   # prepare() gets the union of the other regions as `mask`
    allow_bg = False    # needs_mask restorers usable on the background (mask = outside the regions)
    edits_photo = False # on a region: a removal/replacement that changes the photo itself; the
                        # background and the other regions are then processed on the edited photo
    prompt_hint = None  # placeholder text if the restorer takes a text prompt (values['prompt'])

    @classmethod
    def available(cls, weights_dir):
        """False if the restorer's local weights are missing (it is then not offered)."""
        files = getattr(cls, 'weight_files', ())
        return all(os.path.isfile(os.path.join(weights_dir, f)) for f in files)

    def prepare(self, img: np.ndarray, mask: np.ndarray = None):
        raise NotImplementedError

    def render(self, state, values: dict) -> np.ndarray:
        raise NotImplementedError

    def render_many(self, state, values_list) -> list:
        """Several control settings at once; restorers override this to batch."""
        return [self.render(state, v) for v in values_list]

    def estimate(self, state) -> dict:
        """Blind estimate of the controls for this crop, if the model has one."""
        return {}

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def spec(self) -> dict:
        return {'key': self.key, 'label': self.label, 'description': self.description,
                'params': [asdict(p) for p in self.params], 'group': self.group,
                'needs_mask': self.needs_mask, 'blend': self.blend, 'bg_only': self.bg_only,
                'uses_foreground': self.uses_foreground, 'allow_bg': self.allow_bg, 'prompt': self.prompt_hint,
                'edits': self.edits_photo}


class Identity(Restorer):
    key = 'none'
    label = 'Original'
    description = 'Leave this region untouched.'
    params = []

    def prepare(self, img, mask=None):
        return img

    def render(self, state, values):
        return state
