# --------------------------------------------
# End-to-end test: edit layers
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Removals change the photo the other layers work on: remove a person, then brighten the
background; the removed area must be brightened like the rest (no dark box)."""
import io, json, os, urllib.request
import numpy as np
from PIL import Image
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
def call(path, body=None, method=None, headers=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(RAP_URL + path, data=data, method=method or ('POST' if data is not None else 'GET'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = r.read()
        out = json.loads(out) if r.headers['content-type'].startswith('application/json') else out
        return (out, r.headers.get('X-Base')) if headers else out
def export(sid):
    return np.asarray(Image.open(io.BytesIO(call(f'/api/session/{sid}/export?fmt=png'))).convert('RGB')).astype(np.float32)

s = call('/api/session/example/park_shadows.jpg', {}); sid = s['id']
assert s['base_tag'] == 'orig'
r = call(f'/api/session/{sid}/text', {'prompt': 'person', 'separate': False})['regions'][0]
_, tag = call(f'/api/session/{sid}/region/{r["id"]}/sweep', {'restorer': 'objectclear'}, headers=True)
assert tag != 'orig', tag
b = call(f'/api/session/{sid}/base'); assert b['url'] and b['tag'] == tag
call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'cidnet'})
both = export(sid)
s2 = call('/api/session/example/park_shadows.jpg', {})['id']
call(f'/api/session/{s2}/region/bg/sweep', {'restorer': 'cidnet'})
bright = export(s2)
orig = np.asarray(Image.open('examples/park_shadows.jpg').convert('RGB')).astype(np.float32)
x0, y0, x1, y1 = [round(v / s['scale']) for v in r['box']]
ratio = both[y0:y1, x0:x1].mean() / bright[y0:y1, x0:x1].mean()
print(f'removal box brightness vs brightened-only: {ratio:.2f} (a dark box would be ~{orig[y0:y1, x0:x1].mean() / bright[y0:y1, x0:x1].mean():.2f})')
assert ratio > 0.85, ratio
changed = np.abs(both - bright).max(-1) > 30
print(f'pixels changed by the removal: {changed.mean() * 100:.1f}% of the photo')
assert changed.mean() > 0.002
call(f'/api/session/{sid}/region/{r["id"]}', {'visible': False}, 'PATCH')
_, tag2 = call(f'/api/session/{sid}', headers=True)
assert tag2 == 'orig', tag2
print('hiding the removal restores the original photo underneath: ok')
print('errors: none')
