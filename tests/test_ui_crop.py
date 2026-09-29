# --------------------------------------------
# End-to-end test: ui crop
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, sys, numpy as np
sys.path.insert(0, '.')
from PIL import Image
from playwright.sync_api import sync_playwright
from rap import geometry
def wait_idle(pg, t=300):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden", timeout=t * 1000); pg.wait_for_timeout(300)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.click('.examples button[title="wright_flyer_1903.jpg"]'); wait_idle(pg)
    pg.keyboard.press('c'); pg.wait_for_timeout(200)
    pg.click('#crop-rotr'); pg.click('#crop-fliph')
    pg.evaluate("() => { const i = document.querySelector('#crop-angle'); i.value = 6; i.dispatchEvent(new Event('input')); }")
    pg.select_option('#crop-ratio', '1'); pg.wait_for_timeout(200)
    pg.screenshot(path='tests/out/ui/90_crop.png')
    st = pg.evaluate("() => ({rot: S.crop.rot, fh: S.crop.flipH, angle: S.crop.angle, rect: S.crop.rect, note: !document.querySelector('#crop-note').hidden})")
    print('crop state', st)
    pg.click('#crop-apply'); wait_idle(pg)
    sid = pg.evaluate('() => S.sess.id')
    got = np.asarray(Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export').body())).convert('RGB')).astype(int)
    src = np.asarray(Image.open('examples/wright_flyer_1903.jpg').convert('RGB'))
    exp, inv = geometry.transform(src, st['rot'], st['fh'], False, st['angle'], st['rect'])
    print('server result', got.shape, 'expected', exp.shape, 'invalid corners', inv is not None,
          '| mean abs diff', round(float(np.abs(got - exp.astype(int)).mean()), 3) if got.shape == exp.shape else 'shape mismatch')
    print('unapply visible', pg.is_visible('#btn-unapply'))
    # second crop: drag the rectangle out past the photo so corners must be filled
    pg.keyboard.press('c'); pg.wait_for_timeout(200)
    pg.evaluate("() => { const i = document.querySelector('#crop-angle'); i.value = -10; i.dispatchEvent(new Event('input')); S.crop.rect = [0, 0, 1, 1]; drawOverlay(); }")
    print('note shown', pg.is_visible('#crop-note'))
    pg.click('#crop-apply'); wait_idle(pg)
    sid = pg.evaluate('() => S.sess.id')
    got = np.asarray(Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export').body())).convert('RGB'))
    Image.fromarray(got).save('tests/out/crop_filled.jpg', quality=90)
    print('filled result', got.shape, 'corner pixels', got[2, 2], got[-3, -3])
    b.close()
print('errors:', errs or 'none')
