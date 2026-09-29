# --------------------------------------------
# End-to-end test: ui bg
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, base64, time, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=600):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(500)
def canvas_rgba(pg):
    url = pg.evaluate("() => canvas.toDataURL('image/png')")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).convert('RGBA')).astype(int)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    home = lambda f: (pg.goto(RAP_URL + '/'), pg.wait_for_selector('.examples button'), pg.click(f'.examples button[title="{f}"]'), wait_idle(pg))
    # 1) remove background
    home('bird_blur.png')
    t = time.time(); pg.click('#btn-removebg'); wait_idle(pg); print('remove bg', round(time.time() - t, 1), 's', pg.evaluate("() => S.regions.map(r => [r.name, r.restorer, r.source])"))
    pg.evaluate('() => doneEditing()'); pg.screenshot(path='tests/out/ui/93_removebg.png')
    prev = canvas_rgba(pg)
    sid = pg.evaluate('() => S.sess.id')
    im = Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export?fmt=png').body()))
    exp = np.asarray(im.convert('RGBA')).astype(int)
    print('export', im.mode, im.size, 'alpha range', exp[..., 3].min(), exp[..., 3].max(), '| preview vs export alpha mean diff', round(float(np.abs(prev[..., 3] - exp[..., 3]).mean()), 3),
          'rgb diff where opaque', round(float(np.abs(prev[..., :3] - exp[..., :3])[exp[..., 3] > 250].mean()), 3))
    im.save('tests/out/removebg.png')
    # apply keeps the cut-out
    pg.click('#btn-apply'); wait_idle(pg); print('after apply', pg.evaluate("() => S.regions.map(r => [r.name, r.restorer])"))
    # 2) lens blur with the bench in focus
    home('park_bench.jpg')
    pg.fill('#find-input', 'bench'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    t = time.time(); pg.evaluate("() => chooseRestorer(region('bg'), 'lens_blur')"); wait_idle(pg); print('lens blur', round(time.time() - t, 1), 's')
    pg.evaluate("() => { const i = document.querySelector('.card[data-id=\"bg\"] input[type=range]'); i.value = 0.8; i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); }"); wait_idle(pg)
    pg.evaluate('() => doneEditing()'); pg.screenshot(path='tests/out/ui/94_lensblur.png')
    sid = pg.evaluate('() => S.sess.id')
    Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export?fmt=jpeg').body())).save('tests/out/lensblur.jpg')
    # 3) privacy: pixelate the faces, solid colour background
    home('lunch_1932.jpg')
    pg.fill('#find-input', 'face'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'pixelate')"); wait_idle(pg)
    pg.evaluate("() => chooseRestorer(region('bg'), 'color_fill')"); wait_idle(pg)
    print('menus bg has transparent:', pg.evaluate("() => offeredFor(region('bg'), 'transparent')"),
          '| region has transparent:', pg.evaluate(f"() => offeredFor(region('{rid}'), 'transparent')"))
    pg.evaluate('() => doneEditing()'); pg.screenshot(path='tests/out/ui/95_privacy.png')
    b.close()
print('errors:', errs or 'none')
