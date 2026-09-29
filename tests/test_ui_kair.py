# --------------------------------------------
# End-to-end test: ui kair
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import os
from playwright.sync_api import sync_playwright
OUT = 'tests/out/ui_kair'; os.makedirs(OUT, exist_ok=True)
errors = []
def wait_idle(pg, t=180):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(300)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'{m.type}: {m.text}') if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.wait_for_timeout(500)
    pg.click('.examples button[title="building_noise.png"]'); wait_idle(pg)
    pg.wait_for_function("() => !document.querySelector('#suggest-status').textContent", timeout=120000)
    print('suggestions:', pg.inner_text('#suggest-list').replace('\n', ' | ')[:300])
    pg.click('.card[data-id="bg"] .pick-btn')
    pg.wait_for_timeout(400); pg.screenshot(path=f'{OUT}/01_picker.png')
    pg.evaluate("() => closePicker()")
    pg.evaluate("() => chooseRestorer(region('bg'), 'scunet')"); wait_idle(pg)
    print('scunet values', pg.evaluate("region('bg').values"), 'cands', pg.evaluate("region('bg').cands.length"))
    pg.screenshot(path=f'{OUT}/02_scunet.png')
    pg.evaluate("() => chooseRestorer(region('bg'), 'drunet')"); wait_idle(pg)
    print('drunet values', pg.evaluate("region('bg').values"), 'cands', pg.evaluate("region('bg').cands.length"))
    pg.screenshot(path=f'{OUT}/03_drunet.png')
    b.close()
print('errors:', errors)
