# --------------------------------------------
# Unit tests: adjust
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import numpy as np

from rap import adjust


def img(seed=0, h=48, w=64):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_defaults_are_identity():
    a = img()
    assert adjust.is_identity({}, has_restorer=False)
    assert np.array_equal(adjust.apply(a, None, {}), a)


def test_strength_blends_restored_with_original():
    a, b = img(0), img(1)
    assert np.array_equal(adjust.apply(a, b, {'strength': 1.0}), b)
    assert np.array_equal(adjust.apply(a, b, {'strength': 0.0}), a)
    half = adjust.apply(a, b, {'strength': 0.5}).astype(int)
    assert np.abs(half - (a.astype(int) + b) / 2).max() <= 1
    assert not adjust.is_identity({'strength': 0.5}, has_restorer=True)


def test_exposure_brightens_and_saturation_zero_is_gray():
    a = img()
    assert adjust.apply(a, None, {'exposure': 1}).mean() > a.mean() + 10
    gray = adjust.apply(a, None, {'saturation': -1}).astype(int)
    assert np.abs(gray[..., 0] - gray[..., 1]).max() <= 2 and np.abs(gray[..., 1] - gray[..., 2]).max() <= 2


def test_every_control_keeps_shape_and_dtype():
    a = img()
    for spec in adjust.SPEC:
        for v in (spec['min'], spec['max']):
            out = adjust.apply(a, None, {spec['key']: v}, full_size=(64, 48))
            assert out.shape == a.shape and out.dtype == np.uint8, spec['key']
