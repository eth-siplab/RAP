# --------------------------------------------
# Unit tests: session
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import time

import numpy as np

from rap.session import Region, Session
from rap.store import SessionStore


def session_with_region():
    s = Session(np.zeros((40, 60, 3), np.uint8), None)
    mask = np.zeros((40, 60), bool)
    mask[5:20, 10:30] = True
    s.push()
    r = s.add_region(Region('thing', mask))
    r.restorer, r.values = 'pisasr', {'sem': 0.8}
    return s, r


def test_undo_redo_restores_regions_and_masks():
    s, r = session_with_region()
    assert s.undo(lambda img: None) and list(s.regions) == ['bg']
    assert s.redo(lambda img: None) and r.id in s.regions
    back = s.regions[r.id]
    assert back.mask.sum() == 15 * 20 and back.values == {'sem': 0.8}


def test_store_round_trip(tmp_path):
    s, r = session_with_region()
    s.img = s.img.copy()
    s.img[0, 0] = (10, 20, 30)
    store = SessionStore(str(tmp_path), log=lambda m: None)
    store.schedule(s)
    for _ in range(100):                      # the write happens in a background thread
        if not store.pending and (tmp_path / s.id / 'state.pkl.z').exists():
            break
        time.sleep(0.05)
    time.sleep(0.2)
    loaded = SessionStore(str(tmp_path), log=lambda m: None).load_all()
    t = loaded[s.id]
    assert t.seg_state is None and np.array_equal(t.img, s.img)
    assert list(t.regions) == ['bg', r.id] and t.regions[r.id].mask.sum() == 300
    assert t.regions[r.id].restorer == 'pisasr' and len(t.history) == 1
    assert Region('new', np.zeros((2, 2), bool)).id != r.id    # ids continue after the loaded ones
