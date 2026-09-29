# --------------------------------------------
# End-to-end test: nav
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
OUT = 'tests/out/ui_nav'; os.makedirs(OUT, exist_ok=True)
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
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'{m.type}: {m.text}') if m.type == 'error' else None)
    pg.on('dialog', lambda d: d.accept())
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.wait_for_timeout(500)
    pg.click('.examples button[title="bird_blur.png"]'); wait_idle(pg)
    print('modehint point:', pg.is_visible('#modehint'), pg.inner_text('#modehint')[:60])
    x, y = img_to_screen(pg, 400, 300); pg.mouse.click(x, y); wait_idle(pg)
    print('after click: editing', pg.evaluate('S.editing'), 'chip', pg.is_visible('#editing'))
    pg.screenshot(path=f'{OUT}/01_selected.png')
    # click beside the image -> deselect
    ox, oy = img_to_screen(pg, 0, 0)
    ex, ey = img_to_screen(pg, pg.evaluate('canvas.width'), pg.evaluate('canvas.height'))
    vp = pg.locator('#viewport').bounding_box()
    print('image on screen', [round(v) for v in (ox, oy, ex, ey)], 'viewport', vp)
    pg.mouse.click((ex + vp['x'] + vp['width']) / 2, (oy + ey) / 2)
    pg.wait_for_timeout(300)
    print('after outside click: editing', pg.evaluate('S.editing'), 'chip', pg.is_visible('#editing'), 'regions', pg.evaluate("S.regions.length"))
    # reselect via card, then delete from chip
    rid = pg.evaluate("S.regions.find(r => r.id !== 'bg').id")
    pg.evaluate(f"setActive('{rid}')"); pg.wait_for_timeout(200)
    pg.click('#btn-del-region'); wait_idle(pg)
    print('after chip delete: regions', pg.evaluate("S.regions.map(r=>r.id)"), 'chip', pg.is_visible('#editing'))
    # pan mode hint + back button
    pg.keyboard.press('h'); pg.wait_for_timeout(200)
    print('pan hint:', pg.inner_text('#modehint')[:80]); pg.screenshot(path=f'{OUT}/02_pan.png')
    pg.click('#mh-back'); pg.wait_for_timeout(200)
    print('after back: tool', pg.evaluate('S.tool'))
    # home
    pg.click('#btn-home'); pg.wait_for_timeout(500)
    print('home: empty visible', pg.is_visible('#empty'), 'toolbar', pg.is_visible('#toolbar'), 'hash', repr(pg.evaluate('location.hash')), 'auto disabled', pg.is_disabled('#btn-auto'))
    pg.screenshot(path=f'{OUT}/03_home.png')
    pg.click('.examples button[title="faces_old.jpg"]') if pg.locator('.examples button[title="faces_old.jpg"]').count() else pg.click('.examples button >> nth=1')
    wait_idle(pg)
    print('reopen: toolbar', pg.is_visible('#toolbar'), 'modehint', pg.is_visible('#modehint'), 'auto enabled', not pg.is_disabled('#btn-auto'))
    pg.screenshot(path=f'{OUT}/04_reopen.png')
    b.close()
print('errors:', errors)
