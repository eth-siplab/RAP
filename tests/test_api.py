# --------------------------------------------
# End-to-end test: api
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import json, time, urllib.request
B = RAP_URL
def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(B + path, data=data, method=method or ('POST' if data is not None else 'GET'),
                                 headers={'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = r.read()
    return (json.loads(out) if r.headers['content-type'].startswith('application/json') else out), time.time() - t
s, dt = call('/api/session/example/bird_blur.png', {}); sid = s['id']; print('session', s['width'], s['height'], s['preview'], f'{dt:.2f}s')
t, dt = call(f'/api/session/{sid}/text', {'prompt': 'bird', 'separate': False}); r = t['regions'][0]; print('text', t['count'], r['id'], r['box'], r['score'], f'{dt:.2f}s')
c, dt = call(f'/api/session/{sid}/click', {'region_id': None, 'points': [[640, 120, 1]]}); print('click', c['id'], c['box'], round(c['score'], 3), f'{dt:.2f}s')
sw, dt = call(f'/api/session/{sid}/region/{r["id"]}/sweep', {'restorer': 'pisasr'}); print('sweep pisa', len(sw['candidates']), sw['values'], f'{dt:.2f}s')
sw2, dt = call(f'/api/session/{sid}/region/{r["id"]}/sweep', {'restorer': 'pisasr', 'values': sw['values']}); print('sweep pisa again (cached)', f'{dt:.2f}s')
bg, dt = call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'drunet'}); print('sweep bg drunet', bg['values'], bg['estimate'], f'{dt:.2f}s')
rr, dt = call(f'/api/session/{sid}/region/{r["id"]}/render', {'values': {'sem': 0.9, 'pix': 1.0}}); print('render', rr['url'], f'{dt:.2f}s')
p, dt = call(f'/api/session/{sid}/region/{r["id"]}', {'feather': 4}, 'PATCH'); print('patch feather', p['feather'], f'{dt:.2f}s')
png, dt = call(f'/api/session/{sid}/export'); open('tests/out/export_bird.png', 'wb').write(png); print('export', len(png), f'{dt:.2f}s')
