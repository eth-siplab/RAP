# --------------------------------------------
# End-to-end test: ui
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import sys, time
from playwright.sync_api import sync_playwright
OUT = 'tests/out/ui'
import os; os.makedirs(OUT, exist_ok=True)
errors = []
def wait_idle(pg, t=120):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(300)
def img_to_screen(pg, ix, iy):
    return pg.evaluate(f"""() => {{ const r = document.querySelector('#viewport').getBoundingClientRect();
        return [r.left + S.view.x + {ix} * S.view.k, r.top + S.view.y + {iy} * S.view.k]; }}""")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
    pg.on('console', lambda m: errors.append(f'{m.type}: {m.text}') if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.wait_for_timeout(800)
    pg.screenshot(path=f'{OUT}/01_empty.png')
    pg.click('.examples button[title="bird_blur.png"]'); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/02_opened.png')
    x, y = img_to_screen(pg, 400, 300); pg.mouse.click(x, y); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/03_clicked.png')
    rid = pg.evaluate("() => S.active")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'pisasr')"); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/04_pisa.png')
    pg.click('#btn-done'); pg.wait_for_timeout(200)
    pg.evaluate("() => chooseRestorer(region('bg'), 'nafnet_deblur')"); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/05_bg_deblur.png')
    # drag the primary slider of the bird region to 0 (preview) then release
    pg.evaluate(f"""() => {{ const i = document.querySelector('.card[data-id="{rid}"] input[type=range]'); i.value = 0;
        i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); }}"""); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/06_detail0.png')
    pg.click('#btn-split'); pg.wait_for_timeout(400)
    pg.screenshot(path=f'{OUT}/07_split.png')
    pg.click('#btn-split')
    # text prompt on a fresh image
    pg.click('#zoom-fit')
    pg.fill('#find-input', 'water'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    pg.screenshot(path=f'{OUT}/08_text.png')
    b.close()
print('\n'.join(errors) or 'no console errors')
