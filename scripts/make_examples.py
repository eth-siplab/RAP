# --------------------------------------------
# Example photos from the 2023 RAP demos
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Build the degraded example images from the clean originals of the 2023 RAP demos.

    python scripts/make_examples.py /path/to/RAP/testsets
"""
import io
import os
import sys

import numpy as np
from PIL import Image

src = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)


def load(p):
    return Image.open(os.path.join(src, p)).convert('RGB')


def noise(img, sigma):
    a = np.asarray(img).astype(np.float32) + rng.normal(0, sigma, (img.height, img.width, 3))
    return Image.fromarray(a.round().clip(0, 255).astype(np.uint8))


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=q)
    return Image.open(buf).convert('RGB')


noise(load('build/build.png'), 25).save(f'{out}/building_noise.png')
noise(load('coffee/coffee.png'), 30).save(f'{out}/coffee_noise.png')
jpeg(load('stone/IMG_9919.png'), 8).save(f'{out}/ruin_jpeg.jpg', quality=100)
jpeg(load('van/van.png'), 12).save(f'{out}/sign_jpeg.jpg', quality=100)
