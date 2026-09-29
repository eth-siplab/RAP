# --------------------------------------------
# End-to-end test: ui human
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import time
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=600):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(500)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.wait_for_timeout(1500)
    print('chip:', pg.text_content('#models-text'))
    pg.click('.examples button[title="lunch_1932.jpg"]'); wait_idle(pg)
    pg.wait_for_function("() => document.querySelectorAll('.sg-card').length > 0 || document.querySelector('#suggest').hidden", timeout=60000)
    print('suggestions:', pg.evaluate("() => [...document.querySelectorAll('.sg-title')].map(e => e.textContent)"))
    print('steps done:', pg.evaluate("() => [...document.querySelectorAll('#steps li.done')].map(e => e.dataset.step)"))
    pg.screenshot(path='tests/out/ui/a0_open.png')
    # picker
    pg.click('.card[data-id="bg"] .pick-btn'); pg.wait_for_timeout(300)
    pg.screenshot(path='tests/out/ui/a1_picker.png')
    print('picker items:', pg.evaluate("() => [...document.querySelectorAll('#picker .pk-item .pk-name')].map(e => e.textContent).join(' | ')"))
    pg.click('#picker .pk-item:has-text("Enhance photo")'); wait_idle(pg)
    print('bg restorer', pg.evaluate("() => region('bg').restorer"), 'thumbnails', pg.evaluate("() => document.querySelectorAll('.card[data-id=\"bg\"] .filmstrip img').length"))
    pg.click('.card[data-id="bg"] .filmstrip img >> nth=2'); wait_idle(pg)
    print('after thumb click values', pg.evaluate("() => region('bg').values"), 'steps', pg.evaluate("() => [...document.querySelectorAll('#steps li.done')].map(e => e.dataset.step)"))
    # suggestion "faces" -> auto enhance
    pg.click('.sg-card:has-text("faces") .btn'); wait_idle(pg)
    print('after suggestion:', pg.evaluate("() => S.regions.map(r => r.name + ':' + r.restorer)"))
    pg.hover('.card >> nth=0'); pg.wait_for_timeout(300); pg.screenshot(path='tests/out/ui/a2_hover.png')
    pg.click('#models-chip'); pg.wait_for_timeout(500); pg.screenshot(path='tests/out/ui/a3_models.png'); pg.click('#models-chip')
    # reload keeps the session
    url = pg.url; pg.reload(); pg.wait_for_function("() => S.sess && document.querySelector('#busy').hidden", timeout=120000); wait_idle(pg)
    print('after reload:', url.split('#')[1], pg.evaluate("() => S.regions.map(r => r.name + ':' + r.restorer + ':' + (!!r.display))"))
    b.close()
print('errors:', errs or 'none')
