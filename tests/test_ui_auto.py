# --------------------------------------------
# End-to-end test: ui auto
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
from playwright.sync_api import sync_playwright
errs, toasts = [], []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button')
    pg.set_input_files('#file2', 'third_party/CodeFormer/inputs/whole_imgs/00.jpg')
    pg.wait_for_function("() => S.orig && document.querySelector('#busy').hidden", timeout=120000)
    pg.evaluate("() => { const t = document.querySelector('#toast'); new MutationObserver(() => window.__toasts = (window.__toasts || []).concat([t.textContent])).observe(t, {childList: true, characterData: true, subtree: true}); }")
    pg.evaluate("() => { window.__autoDone = false; autoEnhance().then(() => window.__autoDone = 'ok', e => window.__autoDone = 'err: ' + e.message); }")
    pg.wait_for_function("() => window.__autoDone", timeout=600000)
    print('auto:', pg.evaluate("() => window.__autoDone"))
    print(pg.evaluate("() => S.regions.map(r => [r.id, r.name, r.restorer, r.values, r.info || ''])"))
    print('toasts:', pg.evaluate("() => window.__toasts"))
    b.close()
print('errors:', errs or 'none')
