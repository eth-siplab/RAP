# --------------------------------------------
# End-to-end test: ui flows
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, time, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=900):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(500)
errs = []
def shot(pg, name):
    pg.evaluate("() => doneEditing()"); pg.wait_for_timeout(200); pg.screenshot(path=f'tests/out/ui/{name}.png')
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    home = lambda: (pg.goto(RAP_URL + '/'), pg.wait_for_selector('.examples button'))
    # --- old photo: colorize -> apply -> auto enhance -> export 2x
    home(); pg.click('.examples button[title="lunch_1932.jpg"]'); wait_idle(pg)
    t = time.time(); pg.evaluate("() => chooseRestorer(region('bg'), 'ddcolor')"); wait_idle(pg); print('colorize', round(time.time() - t, 1), 's')
    shot(pg, '70_lunch_color')
    pg.click('#btn-apply'); wait_idle(pg); print('applied; unapply visible:', pg.is_visible('#btn-undo-all'), 'regions', pg.evaluate("() => S.regions.length"))
    t = time.time(); pg.click('#btn-auto'); wait_idle(pg); print('auto', round(time.time() - t, 1), 's', pg.evaluate("() => S.regions.map(r => [r.name, r.restorer, r.info || ''])"))
    shot(pg, '71_lunch_auto')
    pg.click('#btn-undo-all'); wait_idle(pg); print('unapplied; regions', pg.evaluate("() => S.regions.length"), 'unapply visible', pg.is_visible('#btn-undo-all'))
    pg.click('#btn-undo-all') if pg.is_visible('#btn-undo-all') else None
    # re-apply for the export check
    pg.evaluate("() => chooseRestorer(region('bg'), 'ddcolor')"); wait_idle(pg); pg.click('#btn-apply'); wait_idle(pg); pg.click('#btn-auto'); wait_idle(pg)
    sid = pg.evaluate('() => S.sess.id'); t = time.time()
    r = pg.request.get(f'{RAP_URL}/api/session/{sid}/export?scale=2', timeout=900000)
    im = Image.open(io.BytesIO(r.body())); im.save('tests/out/lunch_final_x2.png'); print('export x2', im.size, round(time.time() - t, 1), 's')
    # --- object removal: park bench
    home(); pg.click('.examples button[title="park_bench.jpg"]'); wait_idle(pg)
    pg.fill('#find-input', 'bench'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    t = time.time(); pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'objectclear')"); wait_idle(pg); print('objectclear', round(time.time() - t, 1), 's')
    shot(pg, '72_bench_removed')
    pg.keyboard.down('Space'); pg.wait_for_timeout(200); pg.screenshot(path='tests/out/ui/72b_bench_orig.png'); pg.keyboard.up('Space')
    # --- low light
    home(); pg.click('.examples button[title="night_streetlight.jpg"]'); wait_idle(pg)
    pg.evaluate("() => chooseRestorer(region('bg'), 'cidnet')"); wait_idle(pg); shot(pg, '73_night')
    b.close()
print('errors:', errs or 'none')
