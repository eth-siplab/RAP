# --------------------------------------------
# End-to-end test: ui remove
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=600):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(400)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.goto(RAP_URL + '/'); pg.wait_for_selector('.examples button')
    pg.set_input_files('#file2', 'third_party/ObjectClear/inputs/imgs/test-sample1.jpg'); wait_idle(pg)
    print('preview', pg.evaluate("() => S.sess.preview"), 'scale', pg.evaluate("() => S.sess.scale"))
    pg.fill('#find-input', 'bicycle'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    print('menu (region):', pg.evaluate(f"""() => SECTIONS.map(([t, ks]) => t + ': ' + ks.filter(k => offeredFor(region('{rid}'), k)).join(','))"""))
    print('menu (bg):', pg.evaluate("""() => S.order.filter(k => offeredFor(region('bg'), k)).join(',')"""))
    for key in ['lama', 'objectclear']:
        pg.evaluate(f"() => chooseRestorer(region('{rid}'), '{key}')"); wait_idle(pg)
        pg.evaluate("() => doneEditing()"); pg.screenshot(path=f'tests/out/ui/30_{key}.png')
        print(key, pg.evaluate(f"() => {{ const r = region('{rid}'); return [r.blend, r.box, r.cands.length]; }}"))
    resp = pg.request.get(f"{RAP_URL}/api/session/{pg.evaluate('() => S.sess.id')}/export")
    exp = np.asarray(Image.open(io.BytesIO(resp.body())).convert('RGB'))
    Image.fromarray(exp).save('tests/out/rm_export.jpg', quality=92); print('export', exp.shape)
    b.close()
print('errors:', errs or 'none')
