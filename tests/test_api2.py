# --------------------------------------------
# End-to-end test: api2
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import json, time, urllib.request, uuid
B = RAP_URL
def call(path, body=None, method=None, raw=None, ctype='application/json'):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(B + path, data=data, method=method or ('POST' if data is not None else 'GET'), headers={'Content-Type': ctype})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = r.read()
    return (json.loads(out) if r.headers['content-type'].startswith('application/json') else out), time.time() - t
def upload(path):
    bd = uuid.uuid4().hex
    body = (f'--{bd}\r\nContent-Disposition: form-data; name="file"; filename="x.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + open(path, 'rb').read() + f'\r\n--{bd}--\r\n'.encode()
    return call('/api/session', raw=body, ctype=f'multipart/form-data; boundary={bd}')
s, dt = upload('third_party/CodeFormer/inputs/whole_imgs/00.jpg'); sid = s['id']; print('upload', s['width'], s['height'], f'{dt:.2f}s')
t, dt = call(f'/api/session/{sid}/text', {'prompt': 'face', 'separate': False}); r = t['regions'][0]; print('face regions', t['count'], r['box'], f'{dt:.2f}s')
sw, dt = call(f'/api/session/{sid}/region/{r["id"]}/sweep', {'restorer': 'codeformer'}); print('codeformer sweep', len(sw['candidates']), sw['info'], sw['values'], f'{dt:.2f}s')
bg, dt = call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'pisasr'}); print('bg pisa', f'{dt:.2f}s')
png, dt = call(f'/api/session/{sid}/export'); open('tests/out/export_faces.png', 'wb').write(png); print('export', f'{dt:.2f}s')
s, dt = call('/api/session/example/ruin_jpeg.jpg', {}); sid = s['id']
bg, dt = call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'fbcnn_deblock'}); print('ruin deblock estimate', bg['estimate'], f'{dt:.2f}s')
s, dt = call('/api/session/example/coffee_noise.png', {}); sid = s['id']
bg, dt = call(f'/api/session/{sid}/region/bg/sweep', {'restorer': 'drunet'}); print('coffee drunet estimate', bg['estimate'], '(true sigma 30)', f'{dt:.2f}s')
