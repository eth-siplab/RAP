# --------------------------------------------
# End-to-end test: ui genfill
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import io, time
from PIL import Image
from playwright.sync_api import sync_playwright
def wait_idle(pg, t=900):
    pg.wait_for_timeout(300)
    pg.wait_for_function("() => document.querySelector('#busy').hidden && ![...document.querySelectorAll('.card .status')].some(e => !e.hidden)", timeout=t * 1000)
    pg.wait_for_timeout(500)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    home = lambda f: (pg.goto(RAP_URL + '/'), pg.wait_for_selector('.examples button'), pg.click(f'.examples button[title="{f}"]'), wait_idle(pg))
    home('park_bench.jpg')
    pg.fill('#find-input', 'bench'); pg.press('#find-input', 'Enter'); wait_idle(pg)
    rid = pg.evaluate("() => S.regions[S.regions.length - 1].id")
    pg.evaluate(f"() => chooseRestorer(region('{rid}'), 'genfill')"); pg.wait_for_timeout(300)
    print('waiting for prompt (no server call yet):', pg.evaluate(f"() => [region('{rid}').restorer, region('{rid}').cands.length]"))
    pg.fill(f'.card[data-id="{rid}"] input.prompt', 'a red vintage bicycle, photo')
    t = time.time(); pg.press(f'.card[data-id="{rid}"] input.prompt', 'Enter'); wait_idle(pg); print('generate (incl. model load)', round(time.time() - t, 1), 's')
    print('candidates', pg.evaluate(f"() => region('{rid}').cands.length"), 'values', pg.evaluate(f"() => region('{rid}').values"))
    pg.evaluate("() => { const i = document.querySelector('.card input[type=range]'); i.value = 4; i.dispatchEvent(new Event('input')); i.dispatchEvent(new Event('change')); }"); wait_idle(pg)
    pg.evaluate('() => doneEditing()'); pg.screenshot(path='tests/out/ui/97_genfill.png')
    t = time.time(); pg.fill(f'.card[data-id="{rid}"] input.prompt', 'a large terracotta flower pot with lavender'); pg.press(f'.card[data-id="{rid}"] input.prompt', 'Enter'); wait_idle(pg); print('second prompt', round(time.time() - t, 1), 's')
    sid = pg.evaluate('() => S.sess.id')
    Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export?fmt=jpeg').body())).save('tests/out/genfill_pot.jpg')
    pg.keyboard.press('Control+z'); wait_idle(pg); print('after undo prompt =', pg.evaluate(f"() => region('{rid}').values.prompt"))
    # background replacement
    home('bird_blur.png')
    pg.click('#btn-subject'); wait_idle(pg)
    pg.evaluate("() => chooseRestorer(region('bg'), 'genfill')"); pg.wait_for_timeout(300)
    pg.fill('.card[data-id="bg"] input.prompt', 'a tropical beach at sunset, soft focus, photo')
    t = time.time(); pg.press('.card[data-id="bg"] input.prompt', 'Enter'); wait_idle(pg); print('bg replace', round(time.time() - t, 1), 's')
    pg.evaluate('() => doneEditing()'); pg.screenshot(path='tests/out/ui/98_bg_replace.png')
    sid = pg.evaluate('() => S.sess.id')
    Image.open(io.BytesIO(pg.request.get(f'{RAP_URL}/api/session/{sid}/export?fmt=jpeg').body())).save('tests/out/bg_replace.jpg')
    b.close()
print('errors:', errs or 'none')
