# --------------------------------------------
# README before/after comparisons
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Before/after images for the README results table, made through the running server's API.

    python scripts/make_showcase.py [jpeg] [portrait]      # README comparisons
    python scripts/make_showcase.py pairs [name ...]        # project page before/after pairs
"""
import io
import json
import os
import sys
import urllib.request

from PIL import Image, ImageDraw, ImageFont

RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server to use
H, GAP = 520, 8


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(RAP_URL + path, data=data, method=method or ('POST' if data is not None else 'GET'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = r.read()
        return json.loads(out) if r.headers['content-type'].startswith('application/json') else out


def font(size):
    for f in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(im, text):
    d = ImageDraw.Draw(im)
    f = font(17)
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=f)
    d.rounded_rectangle([x0 - 6, y0 - 5, x1 + 6, y1 + 6], radius=5, fill=(0, 0, 0))
    d.text((12, 10), text, font=f, fill=(255, 255, 255))


def side_by_side(before, after, left_text, right_text, out):
    ims = []
    for im, t in ((before, left_text), (after, right_text)):
        im = im.convert('RGB')
        im = im.resize((round(im.width * H / im.height), H), Image.LANCZOS)
        label(im, t)
        ims.append(im)
    canvas = Image.new('RGB', (ims[0].width + GAP + ims[1].width, H), (22, 24, 28))
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + GAP, 0))
    canvas.save(out, quality=90)
    print('saved', out, canvas.size)


def example(name):
    s = call(f'/api/session/example/{name}', {})
    return s['id'], Image.open(io.BytesIO(call(f'/api/example/{name}')))


def export(sid):
    return Image.open(io.BytesIO(call(f'/api/session/{sid}/export?fmt=png'))).convert('RGB')


def first_region(out):
    return out['regions'][0]['id']


# Each case returns (before, after, note). The README combines them side by side; the
# project page shows them as a draggable before/after pair.
def case_jpeg():
    sid, before = example('parrot_jpeg.jpg')
    qf = call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'fbcnn_deblock'})['values']['qf']
    return before, export(sid), f'FBCNN, quality estimated blindly (≈ {qf:.0f})'


def case_portrait():
    sid, before = example('street_portrait.jpg')
    call(f'/api/session/{sid}/subject', {})
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'lens_blur', 'values': {'amount': 0.8, 'falloff': 0.5}})
    return before, export(sid), 'Select subject, then Blur background (depth)'


def case_old_photo():
    sid, before = example('lunch_1932.jpg')
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'ddcolor'})
    call(f'/api/session/{sid}/apply', {})
    faces = first_region(call(f'/api/session/{sid}/text', {'prompt': 'face', 'separate': False}))
    call(f'/api/session/{sid}/region/{faces}/sweep', {'restorer': 'codeformer'})
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'pisasr'})
    return before, export(sid), 'Colorize, apply, then restore the faces'


def case_removal():
    sid, before = example('park_bench.jpg')
    bench = first_region(call(f'/api/session/{sid}/text', {'prompt': 'bench', 'separate': False}))
    call(f'/api/session/{sid}/region/{bench}/sweep', {'restorer': 'objectclear'})
    return before, export(sid), 'ObjectClear: bench and shadow removed'


def case_generate():
    sid, before = example('bird_blur.png')
    bird = call(f'/api/session/{sid}/subject', {})['id']
    call(f'/api/session/{sid}/region/{bird}/sweep', {'restorer': 'pisasr'})
    call(f'/api/session/{sid}/region/bg/sweep',
         {'restorer': 'genfill', 'values': {'prompt': 'a tropical beach at sunset with palm trees, photo'}})
    return before, export(sid), 'Subject restored, new background (FLUX.2)'


def case_lowlight():
    sid, before = example('night_streetlight.jpg')
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'cidnet', 'values': {'bright': 0.8, 'sat': 0.8}})
    return before, export(sid), 'HVI-CIDNet, brightness 0.8, saturation 0.8'


def case_migrant():
    sid, before = example('migrant_mother_1936.jpg')
    face = first_region(call(f'/api/session/{sid}/text', {'prompt': 'face', 'separate': False}))
    call(f'/api/session/{sid}/region/{face}/sweep', {'restorer': 'codeformer'})
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'pisasr'})
    return before, export(sid), 'Face with CodeFormer, the rest with PiSA-SR'


def case_wright():
    sid, before = example('wright_flyer_1903.jpg')
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'ddcolor'})
    call(f'/api/session/{sid}/apply', {})
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'pisasr'})
    return before, export(sid), 'Colorized with DDColor, then restored with PiSA-SR'


def case_darkpark():
    sid, before = example('park_shadows.jpg')
    person = call(f'/api/session/{sid}/click', {'region_id': None, 'points': [[880, 120, 1]]})['id']
    call(f'/api/session/{sid}/region/{person}/sweep', {'restorer': 'objectclear'})
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'cidnet', 'values': {'bright': 1.0, 'sat': 0.3}})
    return before, export(sid), 'Person removed with ObjectClear, the rest brightened with HVI-CIDNet'


def case_noise():
    sid, before = example('building_noise.png')
    call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'scunet'})
    return before, export(sid), 'Real noise removed with SCUNet'


CASES = {'old_photo': case_old_photo, 'jpeg': case_jpeg, 'removal': case_removal,
         'generate': case_generate, 'portrait': case_portrait, 'lowlight': case_lowlight,
         'migrant': case_migrant, 'wright': case_wright, 'darkpark': case_darkpark, 'noise': case_noise}


FOCUS = {'old_photo': 0.42, 'removal': 0.55, 'generate': 0.55, 'migrant': 0.35, 'wright': 0.5, 'darkpark': 0.45}   # height of the subject, for the 3:2 crop


def crop_3x2(im, fy=0.5, size=(1200, 800)):
    """The same 3:2 frame for every pair, so the comparison grid lines up."""
    W, H = size
    w, h = im.size
    if w / h > W / H:
        nw = round(h * W / H); x0 = (w - nw) // 2; im = im.crop((x0, 0, x0 + nw, h))
    else:
        nh = round(w * H / W); y0 = max(0, min(h - nh, round(h * fy - nh / 2))); im = im.crop((0, y0, w, y0 + nh))
    return im.resize((W, H), Image.LANCZOS)


def pairs(names, out='docs/compare'):
    """Separate before/after images for the project page's comparison sliders."""
    os.makedirs(out, exist_ok=True)
    for n in names:
        before, after, _ = CASES[n]()
        for im, tag in ((before.convert('RGB'), 'before'), (after, 'after')):
            crop_3x2(im, FOCUS.get(n, 0.5)).save(f'{out}/{n}_{tag}.jpg', quality=86)
        print('saved', f'{out}/{n}_before.jpg / _after.jpg')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['pairs']:
        pairs(args[1:] or list(CASES))
    else:
        for n in args or ['jpeg', 'portrait']:
            before, after, note = CASES[n]()
            left = {'jpeg': 'JPEG, quality 10'}.get(n, 'Input')
            side_by_side(before, after, left, note, f'docs/showcase_{n}.jpg')
