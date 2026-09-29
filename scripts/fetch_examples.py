# --------------------------------------------
# Example photos from Wikimedia Commons
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Download the public-domain / CC0 example photos from Wikimedia Commons.

Old photos are downscaled and lightly JPEG-compressed so the face restorer
and upscaler have something to do. Writes examples/*.jpg and records source,
author and licence in examples/examples.json.

    python scripts/fetch_examples.py [file ...]
"""
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
UA = {'User-Agent': 'RestoreAnything-examples/0.2 (open-source research tool)'}

# file name, Commons title, long side, JPEG quality, title shown in the UI, what to try
ITEMS = [
    ('lunch_1932.jpg', 'File:Lunch atop a Skyscraper.jpg', 900, 40,
     'Lunch atop a Skyscraper, 1932', 'Auto enhance restores all eleven faces; try Colorize on the background.'),
    ('park_bench.jpg', 'File:Bench in Xisong Park 20170405a.jpg', 1400, 92,
     'Park bench', 'Click the bench, choose ObjectClear: bench and shadow disappear.'),
    ('park_shadows.jpg', 'File:Lonely Park Shadows (Unsplash).jpg', 1400, 92,
     'Dark park', 'Low light reveals the person on the bench; ObjectClear removes the person with the shadow.'),
    ('migrant_mother_1936.jpg', 'File:Lange-MigrantMother02.jpg', 420, 35,
     'Migrant Mother, 1936', 'Type “face” and restore it with CodeFormer; then Colorize, export at 2×.'),
    ('wright_flyer_1903.jpg', 'File:Wright First Flight 1903Dec17 (full restore 115).jpg', 1000, 60,
     'First flight, 1903', 'Colorize the scene, upscale on export.'),
    ('night_streetlight.jpg', 'File:Street light next to a birch.jpg', 1200, 92,
     'Street light at night', 'Low light at Brightness 0.8 / Saturation 0.8 looks most natural here.'),
    ('street_portrait.jpg', 'File:Stockholm man with headphones (Unsplash).jpg', 1400, 92,
     'Street portrait', 'Select subject, then Blur background: a portrait look from estimated depth.'),
    ('parrot_jpeg.jpg', 'File:Blue, red and green parrot (Unsplash).jpg', 1000, 10,
     'Parrot (JPEG q=10)', 'Fix JPEG artifacts: FBCNN estimates the quality factor by itself.'),
]

EXISTING = {
    'bird_blur.png': ('Seagull (blurred)', 'Click the bird: PiSA-SR at full Detail on it, a gentler Detail on the rest.'),
    'building_noise.png': ('Building (noise)', 'SCUNet removes the grain in one click; DRUNet lets you set the strength.'),
    'coffee_noise.png': ('Coffee (noise)', 'Different denoising strength for the cup and the table.'),
    'ruin_jpeg.jpg': ('Ruin (JPEG q=8)', 'FBCNN JPEG estimates the quality factor by itself.'),
    'sign_jpeg.jpg': ('Sign (tiny JPEG)', 'FBCNN JPEG, then export at 4×.'),
}


def fetch(title, width):
    q = urllib.parse.urlencode({'action': 'query', 'titles': title, 'prop': 'imageinfo',
                                'iiprop': 'url|extmetadata', 'iiurlwidth': width, 'format': 'json'})
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(
                'https://commons.wikimedia.org/w/api.php?' + q, headers=UA), timeout=30))
            ii = next(iter(d['query']['pages'].values()))['imageinfo'][0]
            data = urllib.request.urlopen(urllib.request.Request(ii['thumburl'], headers=UA), timeout=60).read()
            return data, ii
        except Exception as e:  # Commons rate-limits bursts
            print('  retrying:', e)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f'could not fetch {title}')


def main(only=()):
    """Fetch every Commons photo, or only the named files (the other entries are kept)."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'examples.json')
    old = {m['file']: m for m in json.load(open(path))} if os.path.exists(path) else {}
    meta = {}
    for name, commons, side, quality, title, hint in ITEMS:
        if only and name not in only and name in old:
            meta[name] = {**old[name], 'title': title, 'hint': hint}
            continue
        print(name)
        data, ii = fetch(commons, max(side, 1280))
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img.thumbnail((side, side), Image.LANCZOS)
        img.save(os.path.join(OUT, name), quality=quality)
        m = ii['extmetadata']
        artist = re.sub('<[^>]+>', '', m.get('Artist', {}).get('value', '')).strip()
        meta[name] = {'file': name, 'title': title, 'hint': hint, 'source': ii['descriptionurl'],
                      'license': m.get('LicenseShortName', {}).get('value', ''), 'author': artist,
                      'credit': f'{artist or "Unknown"} · {m.get("LicenseShortName", {}).get("value", "")} · Wikimedia Commons'}
        time.sleep(5)
    for name, (title, hint) in EXISTING.items():
        if os.path.exists(os.path.join(OUT, name)):
            meta[name] = {'file': name, 'title': title, 'hint': hint,
                          'credit': 'Restore Anything Pipeline demo images (2023), synthetic degradation'}
    json.dump(list(meta.values()), open(path, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
