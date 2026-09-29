# --------------------------------------------
# End-to-end test: persist
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Sessions survive a server restart: run once with `save`, restart the server, then run with `check`."""
import os
RAP_URL = os.environ.get('RAP_URL', 'http://127.0.0.1:7860')   # server under test
import json, sys, urllib.request
B = RAP_URL
def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(B + path, data=data, method=method or ('POST' if data is not None else 'GET'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), r.headers.get('X-History')
if sys.argv[1] == 'save':
    s, _ = call('/api/session/example/bird_blur.png', {}); sid = s['id']
    c, _ = call(f'/api/session/{sid}/click', {'region_id': None, 'points': [[390, 300, 1]]})
    call(f'/api/session/{sid}/region/{c["id"]}/sweep', {'restorer': 'pisasr'})
    call(f'/api/session/{sid}/region/{c["id"]}', {'feather': 5, 'name': 'bird'}, 'PATCH')
    info, hist = call(f'/api/session/{sid}')
    json.dump({'sid': sid, 'regions': [(r['id'], r['name'], r['restorer'], r['feather']) for r in info['regions']], 'hist': hist},
              open('tests/out/persist.json', 'w'))
    print('saved', sid, [(r['id'], r['name'], r['restorer']) for r in info['regions']], 'history', hist)
else:
    ref = json.load(open('tests/out/persist.json'))
    info, hist = call(f'/api/session/{ref["sid"]}')
    got = [(r['id'], r['name'], r['restorer'], r['feather']) for r in info['regions']]
    print('restored', got, 'history', hist)
    assert got == [tuple(x) for x in ref['regions']], (got, ref['regions'])
    assert hist == ref['hist'], (hist, ref['hist'])
    c, _ = call(f'/api/session/{ref["sid"]}/click', {'region_id': None, 'points': [[100, 100, 1]]})   # SAM state recomputed
    assert c['id'] not in [x[0] for x in ref['regions']], 'region id reused'
    u, _ = call(f'/api/session/{ref["sid"]}/undo', {})
    print('new region', c['id'], 'undo ok, regions', [r['id'] for r in u['regions']])
    print('PERSIST OK')
