# --------------------------------------------
# End-to-end test: adjust2
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, base64, sys, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=300):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(400)
def set_adj(pg, rid, vals):
    pg.evaluate(f"""() => {{ const r = region('{rid}'); r.adjust = Object.assign({{}}, r.adjust, {vals});
        api(`/api/session/${{sid()}}/region/${{r.id}}`, {{ adjust: r.adjust }}, 'PATCH'); renderCard(r); composite(); }}""")
def canvas_px(pg):
    url = pg.evaluate("() => canvas.toDataURL('image/png')")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).convert('RGB')).astype(int)
example = sys.argv[1] if len(sys.argv) > 1 else 'bird_blur.png'
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.click(f'.examples button[title="{example}"]'); wait_idle(pg)
    print('scale', pg.evaluate('() => S.sess.scale'))
    set_adj(pg, 'bg', "{exposure: 0.3, contrast: 0.1, highlights: -0.6, shadows: 0.5, whites: 0.3, blacks: -0.3, temperature: 0.2, tint: -0.2, vibrance: 0.4, saturation: 0.1, clarity: 0.6, dehaze: 0.4, sharpen: 0.5, vignette: 0.5}")
    wait_idle(pg)
    xy = pg.evaluate("() => { const r = viewport.getBoundingClientRect(); return [r.left + S.view.x + canvas.width*0.5*S.view.k, r.top + S.view.y + canvas.height*0.5*S.view.k]; }")
    pg.mouse.click(*xy); wait_idle(pg)
    rid = pg.evaluate('() => S.editing')
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'pisasr')"); wait_idle(pg)
    set_adj(pg, rid, "{strength: 0.7, shadows: -0.4, vibrance: -0.5, clarity: -0.5, dehaze: -0.3, vignette: -0.6, tint: 0.4}")
    wait_idle(pg); pg.wait_for_timeout(600)
    prev = canvas_px(pg)
    sid = pg.evaluate('() => S.sess.id')
    exp = np.asarray(Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export').body())).convert('RGB'))
    if exp.shape[:2] != prev.shape[:2]:
        exp = np.asarray(Image.fromarray(exp).resize((prev.shape[1], prev.shape[0]), Image.LANCZOS))
    d = np.abs(prev - exp.astype(int))
    print('preview vs export: mean', round(d.mean(), 3), 'p99', np.percentile(d, 99), 'max', d.max())
    # auto buttons
    pg.evaluate("() => doneEditing()"); pg.click('.card[data-id="bg"] summary')
    pg.click('.card[data-id="bg"] .adj-actions button:has-text("Auto tone")'); pg.click('.card[data-id="bg"] .adj-actions button:has-text("Auto color")')
    pg.wait_for_timeout(500)
    print('bg adjust after auto:', {k: v for k, v in pg.evaluate("() => region('bg').adjust").items() if k in ('exposure', 'whites', 'blacks', 'temperature', 'tint')})
    pg.screenshot(path='tests/out/ui/91_adjust_panel.png')
    b.close()
print('errors:', errs or 'none')
