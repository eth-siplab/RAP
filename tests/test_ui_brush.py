# --------------------------------------------
# End-to-end test: ui brush
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
AREA = """() => { const r = region(S.editing || S.active); if (!r || !r.overlayImg) return 0;
  const c = document.createElement('canvas'); c.width = r.overlayImg.width; c.height = r.overlayImg.height;
  const t = c.getContext('2d'); t.drawImage(r.overlayImg, 0, 0); const d = t.getImageData(0, 0, c.width, c.height).data;
  let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 127; return n; }"""
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.click('.examples button[title="bird_blur.png"]'); wait_idle(pg)
    scr = lambda ix, iy: pg.evaluate(f"() => {{ const r = viewport.getBoundingClientRect(); return [r.left + S.view.x + {ix}*S.view.k, r.top + S.view.y + {iy}*S.view.k]; }}")
    pg.keyboard.press('p')
    def drag(pts, button='left'):
        x, y = scr(*pts[0]); pg.mouse.move(x, y); pg.mouse.down(button=button)
        for q in pts[1:]:
            x, y = scr(*q); pg.mouse.move(x, y, steps=8)
        pg.mouse.up(button=button); wait_idle(pg)
    drag([(100, 450), (300, 450)]); a1 = pg.evaluate(AREA); n1 = pg.evaluate("() => S.regions.length")
    pg.screenshot(path='tests/out/ui/40_brush.png')
    drag([(100, 500), (300, 500)]); a2 = pg.evaluate(AREA)
    drag([(150, 440), (150, 510)], button='right'); a3 = pg.evaluate(AREA)
    pg.keyboard.press('Control+z'); wait_idle(pg); a4 = pg.evaluate(AREA)
    pg.keyboard.press('v'); x, y = scr(400, 300); pg.mouse.click(x, y); wait_idle(pg); a5 = pg.evaluate(AREA)
    print('regions', n1, '| areas: stroke1', a1, 'stroke2', a2, 'erase', a3, 'undo', a4, 'after SAM click (bird + strokes)', a5)
    ok = a2 > a1 and a3 < a2 and a4 == a2 and a5 > a2
    print('OK' if ok else 'FAIL')
    pg.screenshot(path='tests/out/ui/41_brush_click.png')
    b.close()
print('errors:', errs or 'none')
