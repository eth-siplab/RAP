# --------------------------------------------
# Restorer registry
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

from .base import Identity, Param, Restorer


def load_restorers(weights_dir, device, log=lambda m: print(m, flush=True), depth_model=None):
    """Instantiate every available restorer, in the order shown in the UI."""
    from .cidnet import CIDNetLowLight
    from .codeformer import CodeFormer
    from .ddcolor import DDColorRestorer
    from .effects import Blur, LensBlur, Pixelate, SolidColor, Transparent
    from .fbcnn import FBCNNDeblock
    from .genfill import GenFill
    from .kair import DRUNetDenoise, SCUNetReal
    from .lama import LaMa
    from .nafnet import NAFNetDeblur
    from .objectclear import ObjectClear
    from .pisasr import PiSASR

    out = {'none': Identity()}
    for cls in (PiSASR, SCUNetReal, DRUNetDenoise, NAFNetDeblur, FBCNNDeblock, CodeFormer, LaMa, ObjectClear,
                DDColorRestorer, CIDNetLowLight, GenFill):
        try:
            log(f'loading {cls.label} ...')
            out[cls.key] = cls(weights_dir, device)
        except Exception as e:  # keep the app usable if one model is missing
            log(f'  skipped {cls.key}: {type(e).__name__}: {e}')
    for r in (Blur(), Pixelate(), SolidColor(), Transparent()):
        out[r.key] = r
    if depth_model is not None:
        out['lens_blur'] = LensBlur(depth_model)
    for r in out.values():
        # Region masks are sent to the browser cropped with the default margin,
        # so restorers that composite through the mask must use that margin.
        assert r.blend == 'box' or r.context == 0.15, r.key
    return out


def _classes():
    from .cidnet import CIDNetLowLight
    from .codeformer import CodeFormer
    from .ddcolor import DDColorRestorer
    from .fbcnn import FBCNNDeblock
    from .genfill import GenFill
    from .kair import DRUNetDenoise, SCUNetReal
    from .lama import LaMa
    from .nafnet import NAFNetDeblur
    from .objectclear import ObjectClear
    from .pisasr import PiSASR
    return (PiSASR, SCUNetReal, DRUNetDenoise, NAFNetDeblur, FBCNNDeblock, CodeFormer, LaMa, ObjectClear,
            DDColorRestorer, CIDNetLowLight, GenFill)


def build_restorers(weights_dir, device, manager, log=lambda m: print(m, flush=True)):
    """Restorers as lazy proxies registered with the model manager (nothing is loaded here)."""
    from ..models import LazyRestorer
    from .effects import Blur, LensBlur, Pixelate, SolidColor, Transparent

    out = {'none': Identity()}
    for cls in _classes():
        if not cls.available(weights_dir):
            log(f'{cls.label}: weights not found, not offered')
            continue
        manager.register(cls.key, cls.label, lambda cls=cls: cls(weights_dir, device))
        out[cls.key] = LazyRestorer(cls, manager)
    for r in (Blur(), Pixelate(), SolidColor(), Transparent(), LensBlur(lambda img: manager.get('depth')(img))):
        out[r.key] = r
    for r in out.values():
        assert r.blend != 'mask' or r.context == 0.15, r.key
    return out
