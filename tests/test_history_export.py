# --------------------------------------------
# End-to-end test: history export
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=300):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(300)
state = lambda pg: pg.evaluate("() => ({n: S.regions.length, regs: S.regions.map(r => r.id + ':' + r.restorer + (adjIdentity({...r, restorer: 'none'}) ? '' : '+adj')).join(' '), w: S.sess.width, undo: !document.querySelector('#btn-undo-all').disabled, redo: !document.querySelector('#btn-redo-all').disabled})")
# A small JPEG with camera EXIF and a rotation tag, to check that export keeps metadata.
import os
os.makedirs('tests/out/ui', exist_ok=True)
if not os.path.exists('tests/out/exif_test.jpg'):
    ex = Image.Exif(); ex[0x010F] = 'TestCam'; ex[0x0110] = 'Model X'; ex[0x0112] = 6
    Image.open('examples/bird_blur.png').convert('RGB').resize((600, 800)).save('tests/out/exif_test.jpg', exif=ex)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, accept_downloads=True)
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button'); pg.click('.examples button[title="bird_blur.png"]'); wait_idle(pg)
    log = [('open', state(pg))]
    pg.evaluate("() => chooseRestorer(region('bg'), 'pisasr')"); wait_idle(pg); log.append(('bg pisa', state(pg)))
    xy = pg.evaluate("() => { const r = viewport.getBoundingClientRect(); return [r.left + S.view.x + 400*S.view.k, r.top + S.view.y + 300*S.view.k]; }")
    pg.mouse.click(*xy); wait_idle(pg); log.append(('click region', state(pg)))
    rid = pg.evaluate('() => S.editing')
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'fbcnn_deblock')"); wait_idle(pg); log.append(('region fbcnn', state(pg)))
    pg.evaluate(f"() => {{ const r = region('{rid}'); r.adjust = {{exposure: 0.5}}; api(`/api/session/${{sid()}}/region/${{r.id}}`, {{adjust: r.adjust}}, 'PATCH'); }}"); wait_idle(pg); log.append(('adjust', state(pg)))
    pg.click('#btn-apply'); wait_idle(pg); log.append(('apply', state(pg)))
    pg.keyboard.press('c'); pg.select_option('#crop-ratio', '1'); pg.click('#crop-apply'); wait_idle(pg); log.append(('crop 1:1', state(pg)))
    for i in range(6):
        pg.keyboard.press('Control+z'); wait_idle(pg); log.append((f'undo {i+1}', state(pg)))
    for i in range(2):
        pg.keyboard.press('Control+Shift+z'); wait_idle(pg); log.append((f'redo {i+1}', state(pg)))
    for name, st in log: print(f'{name:14s}', st)
    # export formats
    pg.on('dialog', lambda d: (print('dialog:', d.message), d.accept()))
    pg.set_input_files('#file', 'tests/out/exif_test.jpg'); wait_idle(pg)
    print('uploaded exif image size', pg.evaluate('() => [S.sess.width, S.sess.height]'))
    sid = pg.evaluate('() => S.sess.id')
    for q in ['fmt=png', 'fmt=jpeg&quality=85', 'fmt=webp&quality=80&max_side=400', 'fmt=jpeg&meta=false']:
        r = pg.request.get(f'{RAP_URL}/api/session/{sid}/export?{q}')
        im = Image.open(io.BytesIO(r.body())); ex = im.getexif()
        print(f'{q:36s} {im.format:5s} {im.size} {r.headers.get("x-export-name")} exif orientation={ex.get(0x0112)} make={ex.get(0x010F)}')
    pg.click('#btn-export'); pg.wait_for_timeout(300); pg.screenshot(path='tests/out/ui/92_export_panel.png')
    b.close()
print('errors:', errs or 'none')
