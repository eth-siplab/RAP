# --------------------------------------------
# End-to-end test: adjust
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, base64, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=300):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(400)
def set_slider(pg, rid, label, value):
    pg.evaluate(f"""() => {{ const card = document.querySelector('.card[data-id="{rid}"]');
        const w = [...card.querySelectorAll('.param-top')].find(e => e.querySelector('.lbl').textContent === '{label}').parentElement;
        const i = w.querySelector('input'); i.value = {value}; i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); }}""")
def canvas_px(pg):
    url = pg.evaluate("() => canvas.toDataURL('image/png')")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).convert('RGB')).astype(int)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button')
    pg.click('.examples button[title="bird_blur.png"]'); wait_idle(pg)
    pg.click('.card[data-id="bg"] summary')
    for lab, v in [('Exposure', 0.5), ('Contrast', 0.2), ('Saturation', 0.3), ('Temperature', -0.4), ('Sharpen', 1.0)]:
        set_slider(pg, 'bg', lab, v)
    wait_idle(pg)
    pg.fill('#find-input', 'bird'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'pisasr')"); wait_idle(pg)
    set_slider(pg, rid, 'Strength', 0.5)
    pg.click(f'.card[data-id="{rid}"] summary'); set_slider(pg, rid, 'Exposure', -0.3); set_slider(pg, rid, 'Saturation', 0.5)
    wait_idle(pg); pg.wait_for_timeout(500)
    prev = canvas_px(pg)
    pg.evaluate("() => doneEditing()"); pg.screenshot(path='tests/out/ui/20_adjust.png')
    resp = pg.request.get(f"{RAP_URL}/api/session/{pg.evaluate('() => S.sess.id')}/export")
    exp = np.asarray(Image.open(io.BytesIO(resp.body())).convert('RGB')).astype(int)
    orig = np.asarray(Image.open('examples/bird_blur.png').convert('RGB')).astype(int)
    d = np.abs(prev - exp)
    print('preview vs export: mean', round(d.mean(), 3), 'p99', np.percentile(d, 99), 'max', d.max(), '| export vs original mean', round(np.abs(exp - orig).mean(), 2))
    ys, xs = np.nonzero(d.max(-1) > 3); print('pixels differing by >3:', len(ys))
    print('stored adjust:', pg.evaluate("() => S.regions.map(r => [r.id, r.restorer, r.adjust])"))
    b.close()
print('errors:', errs or 'none')
