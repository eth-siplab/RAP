# --------------------------------------------
# README demo recording
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Record the README demo clips from a running server (python -m rap).

Each demo is driven like a user would: a drawn cursor moves, clicks and types.
Waits for the models are logged so that make_gifs.py can speed them up.
Output: docs/demos/<name>.webm + <name>.json (segments to speed up).

    python scripts/record_demos.py [name ...]
"""
import json
import os
import shutil
import sys
import time

from playwright.sync_api import sync_playwright

RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server to use
URL = RAP_URL + '/'
OUT = 'docs/demos'
W, H = 1280, 800

CURSOR_JS = r"""
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:26px;height:26px;z-index:2147483647;pointer-events:none;transform:translate(-4px,-3px)';
  c.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26"><path d="M4 2 L4 19.5 L8.6 15.3 L11.8 22.2 L15 20.8 L11.8 14 L18.2 14 Z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  document.body.appendChild(c);
  const ring = (x, y, color) => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${x - 18}px;top:${y - 18}px;width:36px;height:36px;border-radius:50%;border:3px solid ${color};z-index:2147483646;pointer-events:none;transition:transform .45s ease-out,opacity .45s ease-out;transform:scale(.3);opacity:1`;
    document.body.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.3)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 600);
  };
  window.addEventListener('pointermove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  window.addEventListener('pointerdown', (e) => ring(e.clientX, e.clientY, e.button === 2 ? '#ff6b6b' : '#ffd84d'), true);
});
"""


class Demo:
    def __init__(self, browser, name):
        self.name = name
        self.tmp = os.path.join(OUT, '_tmp_' + name)
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.ctx = browser.new_context(viewport={'width': W, 'height': H}, record_video_dir=self.tmp,
                                       record_video_size={'width': W, 'height': H})
        self.ctx.add_init_script(CURSOR_JS)
        self.pg = self.ctx.new_page()
        self.t0 = time.time()
        self.fast = []          # [start, end] seconds to speed up
        self.start = 0.0
        self.pos = (W / 2, H / 2)
        self.pg.on('dialog', lambda d: d.accept())

    def now(self):
        return time.time() - self.t0

    def pause(self, s):
        self.pg.wait_for_timeout(int(s * 1000))

    def mark_start(self):
        self.start = self.now()

    def move(self, x, y, dur=0.6):
        steps = max(8, int(dur * 40))
        x0, y0 = self.pos
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)   # ease in-out
            self.pg.mouse.move(x0 + (x - x0) * t, y0 + (y - y0) * t)
            self.pg.wait_for_timeout(int(dur * 1000 / steps))
        self.pos = (x, y)

    def click_xy(self, x, y, button='left', dur=0.6):
        self.move(x, y, dur)
        self.pause(0.15)
        self.pg.mouse.click(x, y, button=button)
        self.pause(0.25)

    def center(self, sel):
        loc = self.pg.locator(sel).first
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        return b['x'] + b['width'] / 2, b['y'] + b['height'] / 2

    def click(self, sel, dur=0.6):
        self.click_xy(*self.center(sel), dur=dur)

    def img_xy(self, ix, iy):
        """Canvas (preview) pixel -> page coordinates."""
        return self.pg.evaluate(f"""() => {{ const r = document.querySelector('#viewport').getBoundingClientRect();
            return [r.left + S.view.x + {ix} * S.view.k, r.top + S.view.y + {iy} * S.view.k]; }}""")

    def type(self, text):
        self.pg.keyboard.type(text, delay=55)

    def idle(self, timeout=900):
        a = self.now()
        self.pg.wait_for_timeout(250)
        self.pg.wait_for_function("() => document.querySelector('#busy').hidden && "
                                  "![...document.querySelectorAll('.card .status')].some(e => !e.hidden)",
                                  timeout=timeout * 1000)
        b = self.now()
        if b - a > 1.2:
            self.fast.append([a + 0.4, b - 0.1])
        self.pause(0.3)

    def slide(self, sel, frac_from, frac_to, dur=1.2):
        loc = self.pg.locator(sel).first
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        y = b['y'] + b['height'] / 2
        x0 = b['x'] + 8 + (b['width'] - 16) * frac_from
        x1 = b['x'] + 8 + (b['width'] - 16) * frac_to
        self.move(x0, y, 0.5)
        self.pg.mouse.down()
        self.move(x1, y, dur)
        self.pg.mouse.up()

    def finish(self):
        end = self.now()
        video = self.pg.video
        self.ctx.close()
        dst = os.path.join(OUT, self.name + '.webm')
        shutil.move(video.path(), dst)
        shutil.rmtree(self.tmp, ignore_errors=True)
        json.dump({'start': self.start, 'end': end, 'fast': self.fast}, open(os.path.join(OUT, self.name + '.json'), 'w'))
        print(f'{self.name}: {end - self.start:.1f} s recorded, {sum(b - a for a, b in self.fast):.1f} s of waits')


def open_example(d, file):
    d.pg.goto(URL)
    d.pg.wait_for_selector('.examples button img')
    d.pause(0.8)
    d.mark_start()
    d.pause(0.4)
    d.click(f'.examples button[title="{file}"]', dur=0.8)
    d.idle()
    d.pause(0.6)


def done(d):
    if d.pg.is_visible('#btn-done'):
        d.click('#btn-done')


def pick(d, rid, name, dur=0.6):
    d.click(f'.card[data-id="{rid}"] .pick-btn', dur=dur)
    d.pause(0.5)
    d.click(f'#picker .pk-item:has(.pk-name:text-is("{name}"))', dur=0.7)


def last_region(d):
    return d.pg.evaluate("() => S.regions[S.regions.length - 1].id")


def split_sweep(d):
    d.click('#btn-split')
    d.pause(0.4)
    x0, y0 = d.img_xy(0, 0)
    x1, y1 = d.img_xy(d.pg.evaluate('canvas.width'), d.pg.evaluate('canvas.height'))
    ym = (y0 + y1) / 2
    xm = (x0 + x1) / 2
    d.move(xm, ym, 0.5)
    d.pg.mouse.down()
    d.move(x0 + (x1 - x0) * 0.12, ym, 1.0)
    d.move(x0 + (x1 - x0) * 0.88, ym, 1.8)
    d.move(xm, ym, 1.0)
    d.pg.mouse.up()
    d.pause(0.8)


# ------------------------------------------------------------------ demos
def caption(d, text):
    """A caption pill over the top of the photo (empty text hides it)."""
    d.pg.evaluate("""(t) => {
        let c = document.getElementById('demo-caption');
        if (!c) {
          c = document.createElement('div'); c.id = 'demo-caption';
          c.style.cssText = 'position:fixed;top:64px;transform:translateX(-50%);z-index:2147483640;padding:9px 20px;' +
            'border-radius:999px;background:rgba(12,14,18,.88);color:#fff;font:600 18px system-ui,sans-serif;' +
            'box-shadow:0 4px 18px rgba(0,0,0,.4);pointer-events:none;white-space:nowrap;transition:opacity .25s';
          document.body.appendChild(c);
        }
        const v = document.querySelector('#viewport').getBoundingClientRect();
        c.style.left = (v.left + v.width / 2) + 'px';
        c.textContent = t; c.style.opacity = t ? 1 : 0; }""", text)


def open_next(d, file):
    """Cut straight to another example (as if opened from the start page)."""
    caption(d, '')
    d.pg.evaluate(f"() => {{ S.dirty = false; return openExample({file!r}); }}")
    d.idle()
    d.pause(0.3)


def deselect(d, ix, iy):
    """Finish the selection (Esc) and rest the cursor on the photo, so no overlay covers the result."""
    d.pg.keyboard.press('Escape')
    d.move(*d.img_xy(ix, iy), 0.4)
    d.pg.evaluate("() => { doneEditing(); S.hoverRegion = null; drawOverlay(); }")
    d.pause(0.2)


def zoom(d, ix, iy, steps=6, dy=-100):
    """Zoom in around a photo point with the mouse wheel."""
    d.move(*d.img_xy(ix, iy), 0.5)
    for _ in range(steps):
        d.pg.mouse.wheel(0, dy)
        d.pause(0.06)
    d.pause(0.4)


def demo_hero(b):
    """Segmentation first: find every face by name, select the subject to blur only the background,
    click a person to remove them with their shadow and brighten the rest."""
    d = Demo(b, 'hero')
    open_example(d, 'migrant_mother_1936.jpg')
    caption(d, 'Type “face”: SAM 3 finds the face → restore only the face')
    d.click('#find-input', dur=0.5)
    d.type('face')
    d.pg.keyboard.press('Enter')
    d.idle()
    d.pause(0.8)
    rid = last_region(d)
    pick(d, rid, 'Restore faces', dur=0.45)
    d.idle()
    deselect(d, 158, 130)
    zoom(d, 158, 130, steps=4)
    for _ in range(2):           # hold Space: before / release: after
        caption(d, 'Before (holding Space)')
        d.pg.keyboard.down(' ')
        d.pause(1.0)
        d.pg.keyboard.up(' ')
        caption(d, 'After: only the face was restored')
        d.pause(1.2)

    open_next(d, 'street_portrait.jpg')
    caption(d, 'Select subject → blur only what is behind them')
    d.click('#btn-subject', dur=0.6)
    d.idle()
    d.pause(0.9)
    done(d)
    pick(d, 'bg', 'Blur background', dur=0.45)
    d.idle()
    deselect(d, 900, 450)
    d.pause(1.0)

    open_next(d, 'park_shadows.jpg')
    caption(d, 'Click the person → remove them with their shadow, brighten the rest')
    d.click_xy(*d.img_xy(880, 120), dur=0.6)
    d.idle()
    d.pause(0.6)
    rid = last_region(d)
    pick(d, rid, 'Remove object + shadow', dur=0.45)
    d.idle()
    done(d)
    pick(d, 'bg', 'Brighten dark photo', dur=0.45)
    d.idle()
    d.pg.wait_for_function("() => S.baseLoaded === S.baseTag", timeout=300000)
    d.idle()
    # the model's colour follows the photo's blue night tint; take the saturation down for neutral tones
    d.slide('.card[data-id="bg"] div:has(> .param-top .lbl:text-is("Saturation")) input[type=range]', 0.425, 0.15, 0.9)
    d.idle()
    deselect(d, 700, 600)
    d.pause(0.5)
    split_sweep(d)
    d.finish()


def demo_select(b):
    """Click an object, choose what to do, slide."""
    d = Demo(b, 'select')
    open_example(d, 'bird_blur.png')
    x, y = d.img_xy(390, 300)
    d.click_xy(x, y, dur=0.9)
    d.idle()
    d.pause(0.5)
    rid = last_region(d)
    pick(d, rid, 'Enhance photo')
    d.idle()
    d.pause(0.5)
    d.slide(f'.card[data-id="{rid}"] input[type=range]', 1.0, 0.0, 1.4)
    d.pause(0.5)
    d.slide(f'.card[data-id="{rid}"] input[type=range]', 0.0, 0.8, 1.2)
    d.idle()
    done(d)
    d.move(*d.img_xy(600, 250), 0.6)
    d.pg.keyboard.down(' ')      # hold Space: original
    d.pause(1.3)
    d.pg.keyboard.up(' ')
    d.pause(1.2)
    d.finish()


def demo_remove(b):
    """Remove an object together with its shadow."""
    d = Demo(b, 'remove')
    open_example(d, 'park_bench.jpg')
    x, y = d.img_xy(*BENCH_PT)
    d.click_xy(x, y, dur=0.9)
    d.idle()
    d.pause(0.5)
    rid = last_region(d)
    pick(d, rid, 'Remove object + shadow')
    d.idle()
    done(d)
    d.pause(0.8)
    split_sweep(d)
    d.finish()


def demo_find(b):
    """Find by name, then generate something new there."""
    d = Demo(b, 'find')
    open_example(d, 'building_noise.png')
    d.click('#find-input')
    d.type('sky')
    d.pause(0.3)
    d.pg.keyboard.press('Enter')
    d.idle()
    d.pause(0.6)
    rid = last_region(d)
    pick(d, rid, 'Replace with…')
    d.pause(0.5)
    d.click(f'.card[data-id="{rid}"] input.prompt')
    d.type('dramatic sunset sky with orange clouds')
    d.pause(0.3)
    d.pg.keyboard.press('Enter')
    d.idle()
    done(d)
    d.pause(0.4)
    pick(d, 'bg', 'Remove noise')
    d.idle()
    d.pause(0.8)
    split_sweep(d)
    d.finish()


BENCH_PT = (0, 0)   # set by probe()
DEMOS = {'hero': demo_hero, 'select': demo_select, 'remove': demo_remove, 'find': demo_find}


def probe(b):
    """Find a click point on the bench (canvas coordinates) without recording."""
    global BENCH_PT
    pg = b.new_page(viewport={'width': W, 'height': H})
    pg.goto(URL)
    pg.wait_for_selector('.examples button')
    pg.click('.examples button[title="park_bench.jpg"]')
    pg.wait_for_function("() => S.sess && document.querySelector('#busy').hidden", timeout=300000)
    pg.fill('#find-input', 'bench')
    pg.press('#find-input', 'Enter')
    pg.wait_for_function("() => S.regions.length > 1 && document.querySelector('#busy').hidden", timeout=300000)
    # the mask pixel nearest the box centre whose 9x9 neighbourhood is all inside the mask
    BENCH_PT = tuple(pg.evaluate("""() => {
        const r = S.regions[S.regions.length - 1], [x0, y0, x1, y1] = r.box, w = x1 - x0, h = y1 - y0;
        const c = document.createElement('canvas'); c.width = w; c.height = h;
        const g = c.getContext('2d'); g.drawImage(r.maskImg, 0, 0, w, h);
        const a = g.getImageData(0, 0, w, h).data, inside = (x, y) => a[(y * w + x) * 4 + 3] > 128;
        let best = null, bd = 1e9;
        for (let y = 4; y < h - 4; y += 2) for (let x = 4; x < w - 4; x += 2) {
          let ok = true;
          for (let dy = -4; dy <= 4 && ok; dy += 2) for (let dx = -4; dx <= 4 && ok; dx += 2) ok = inside(x + dx, y + dy);
          const d = (x - w / 2) ** 2 + (y - h / 2) ** 2;
          if (ok && d < bd) { bd = d; best = [x0 + x, y0 + y]; }
        }
        return best; }"""))
    print('bench click point', BENCH_PT)
    pg.close()


def main():
    names = sys.argv[1:] or list(DEMOS)
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        if 'remove' in names:
            probe(b)
        for n in names:
            DEMOS[n](b)
        b.close()


if __name__ == '__main__':
    main()
