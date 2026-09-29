# --------------------------------------------
# End-to-end test: kair tile
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os, sys; os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); sys.path.insert(0, '.')
import numpy as np, torch, cv2
from PIL import Image
from rap.restorers.kair import SCUNetReal, DRUNetDenoise
img = np.array(Image.open('examples/building_noise.png').convert('RGB'))
img = cv2.resize(img, None, fx=1.6, fy=1.6); print(img.shape)
def psnr(a, b): return 10 * np.log10(255 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))
for cls in (SCUNetReal, DRUNetDenoise):
    r = cls('weights', 'cuda'); v = {r.params[0].key: r.params[0].sweep[2]}
    r.big = 10 ** 9; whole = r.render(r.prepare(img), v)
    r.big = 10 ** 5; tiled = r.render(r.prepare(img), v)
    print(cls.key, 'tiled vs whole', round(psnr(tiled, whole), 1), 'dB')
