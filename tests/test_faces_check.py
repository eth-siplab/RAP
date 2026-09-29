# --------------------------------------------
# End-to-end test: faces check
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=300):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(300)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button')
    pg.set_input_files('#file2', 'third_party/CodeFormer/inputs/whole_imgs/00.jpg'); wait_idle(pg)
    pg.fill('#find-input', 'face'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'codeformer')"); wait_idle(pg)
    pg.evaluate("() => chooseRestorer(region('bg'), 'pisasr')"); wait_idle(pg)
    print(pg.evaluate("() => S.regions.map(r => [r.id, r.restorer, !!r.display, r.info || '', r.cands.length])"))
    print('focused:', pg.evaluate("() => document.activeElement.tagName + '.' + document.activeElement.className + '#' + document.activeElement.id"))
    sig = "() => { const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data; let s = 0; for (let i = 0; i < d.length; i += 97) s += d[i]; return s; }"
    a = pg.evaluate(sig); pg.keyboard.down('Space'); pg.wait_for_timeout(150); c = pg.evaluate(sig); pg.keyboard.up('Space')
    print('restored', a, 'compare', c, 'OK' if a != c else 'FAIL')
    b.close()
