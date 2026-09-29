# --------------------------------------------
# End-to-end test: kair
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os, time, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import numpy as np, torch, cv2
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
sys.path.insert(0, '.')
from rap.restorers.kair import SCUNetReal, DRUNetDenoise, noise_sigma
from PIL import Image
W = 'weights'
def psnr(a, b): return 10 * np.log10(255 ** 2 / np.mean((a.astype(np.float64) - b) ** 2))
clean = np.array(Image.open('examples/building.jpg' if os.path.exists('examples/building.jpg') else sorted(f'examples/{f}' for f in os.listdir('examples') if f.endswith(('.jpg', '.png')))[0]).convert('RGB'))
clean = clean[:768, :1024]
rng = np.random.default_rng(0)
noisy = np.clip(clean + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)
print('input psnr', round(psnr(noisy, clean), 2), 'estimated sigma', round(noise_sigma(noisy), 1), 'clean est', round(noise_sigma(clean), 1))
t = time.time(); sc = SCUNetReal(W, 'cuda'); dr = DRUNetDenoise(W, 'cuda'); print('load', round(time.time() - t, 1), 's')
for name, r in (('scunet', sc), ('drunet', dr)):
    torch.cuda.synchronize(); t = time.time(); st = r.prepare(noisy)
    est = r.estimate(st); vals = [{**r.defaults(), **est}] + [{r.params[0].key: v} for v in r.params[0].sweep]
    outs = r.render_many(st, vals); torch.cuda.synchronize()
    print(name, 'est', est, 'time', round(time.time() - t, 2), 's', 'psnr', [round(psnr(o, clean), 2) for o in outs])
    Image.fromarray(np.concatenate([noisy, outs[0]], 1)).save(f'tests/out/kair_{name}.jpg', quality=92)
# big crop: tiled path
big = cv2.resize(noisy, None, fx=2, fy=2)
for name, r in (('scunet', sc), ('drunet', dr)):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    st = r.prepare(big); o = r.render_many(st, [{r.params[0].key: r.params[0].default}, {r.params[0].key: r.params[0].sweep[-1]}])
    torch.cuda.synchronize(); print(name, 'big', big.shape, round(time.time() - t, 2), 's peak', round(torch.cuda.max_memory_allocated() / 2**30, 2), 'GB')
