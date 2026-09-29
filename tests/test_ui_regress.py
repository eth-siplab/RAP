# --------------------------------------------
# End-to-end test: ui regress
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
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button')
    pg.set_input_files('#file2', 'third_party/CodeFormer/inputs/whole_imgs/00.jpg'); wait_idle(pg)
    pg.fill('#find-input', 'face'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    before = pg.evaluate("() => S.regions.map(r => [r.id, r.source, r.name])")
    xy = pg.evaluate("() => { const r = document.querySelector('#viewport').getBoundingClientRect(); return [r.left + S.view.x + 250*S.view.k, r.top + S.view.y + 280*S.view.k]; }")
    pg.mouse.click(*xy); wait_idle(pg)
    after = pg.evaluate("() => S.regions.map(r => [r.id, r.source, r.name])")
    print('before', before); print('after ', after)
    assert len(after) == len(before) + 1 and after[:len(before)] == before, 'click must start a new region'
    # a second click refines the new region instead of creating another
    pg.mouse.click(xy[0] + 40, xy[1]); wait_idle(pg)
    again = pg.evaluate("() => S.regions.map(r => [r.id, (r.points||[]).length])")
    print('again ', again); assert len(again) == len(after) and again[-1][1] == 2
    pg.click('#btn-done'); pg.mouse.click(xy[0] - 60, xy[1] - 60); wait_idle(pg)
    print('new   ', pg.evaluate("() => S.regions.length")); print('errors', errs or 'none')
    b.close()
