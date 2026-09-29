# --------------------------------------------
# README interface tour
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Screenshot of the app with numbered callouts for the README tour (docs/ui_tour.png).

    python scripts/make_ui_tour.py      # needs a running server
"""
import io
import os

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server to use
URL = RAP_URL + '/'
W, H, DPR = 1440, 900, 2
ACCENT = (255, 196, 0)


def idle(pg):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && "
                         "![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=600000)
    pg.wait_for_timeout(500)


def font(size):
    for f in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': W, 'height': H}, device_scale_factor=DPR)
    pg.goto(URL)
    pg.wait_for_selector('.examples button')
    pg.click('.examples button[title="bird_blur.png"]')
    idle(pg)
    pg.evaluate("() => { try { localStorage.setItem('rap.steps.hidden', '1'); } catch (_) {} updateSteps(); }")
    xy = pg.evaluate("""() => { const r = document.querySelector('#viewport').getBoundingClientRect();
        return [r.left + S.view.x + 390 * S.view.k, r.top + S.view.y + 300 * S.view.k]; }""")
    pg.mouse.click(*xy)
    idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'pisasr')")
    idle(pg)
    pg.evaluate(f"() => chooseRestorer(region('bg'), 'nafnet_deblur')")
    idle(pg)
    pg.evaluate(f"() => setActive('{rid}', false)")
    pg.mouse.move(W / 2, H - 5)
    pg.wait_for_timeout(400)
    boxes = {}
    for key, sel in [('tools', '.toolbar'), ('find', '#find'), ('pick', f'.card[data-id="{rid}"] .pick-btn'),
                     ('film', f'.card[data-id="{rid}"] .filmstrip'), ('compare', '#btn-compare'),
                     ('auto', '#btn-auto'), ('export', '#btn-export'), ('undo', '#btn-undo-all')]:
        boxes[key] = pg.locator(sel).first.bounding_box()
    img = Image.open(io.BytesIO(pg.screenshot())).convert('RGB')
    pg.evaluate("() => { try { localStorage.removeItem('rap.steps.hidden'); } catch (_) {} }")
    b.close()

d = ImageDraw.Draw(img)
f = font(15 * DPR)
callouts = [('tools', 1, 'left'), ('find', 2, 'left'), ('pick', 3, 'left'), ('film', 4, 'left'),
            ('compare', 5, 'bottom'), ('auto', 6, 'bottom'), ('undo', 7, 'bottom'), ('export', 8, 'bottom')]
for key, n, side in callouts:
    bx = boxes[key]
    x0, y0, x1, y1 = [v * DPR for v in (bx['x'], bx['y'], bx['x'] + bx['width'], bx['y'] + bx['height'])]
    d.rounded_rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], radius=8, outline=ACCENT, width=2 * DPR)
    r = 13 * DPR
    cx, cy = (x0 - r - 4, (y0 + y1) / 2) if side == 'left' else ((x0 + x1) / 2, y1 + r + 6)
    cx = max(cx, r + 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ACCENT, outline=(20, 20, 20), width=DPR)
    d.text((cx, cy), str(n), fill=(20, 20, 20), font=f, anchor='mm')
img = img.resize((1200, 750), Image.LANCZOS)
img.save('docs/ui_tour.png', optimize=True)
print('saved docs/ui_tour.png', img.size)
