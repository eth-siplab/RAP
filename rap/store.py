# --------------------------------------------
# Saving sessions to disk
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Keeps sessions on disk so a server restart does not lose anyone's work.

After every change the session's state (image, regions, undo/redo history) is
written in a background thread; on start the newest sessions are loaded back.
Images are stored once per distinct array (history entries share them), the
rest is a compressed pickle. The SAM embedding is not stored: it is recomputed
the first time the session needs it.
"""
import glob
import hashlib
import os
import pickle
import shutil
import threading
import zlib

import cv2
import numpy as np

from .session import Session, reserve_region_ids

HISTORY_KEPT = 20   # undo steps written to disk (the session keeps more in memory)


class SessionStore:
    def __init__(self, root, keep=4, log=print):
        self.root = root
        self.keep = keep
        self.log = log
        self.pending = {}                 # sid -> payload waiting to be written
        self.cond = threading.Condition()
        os.makedirs(root, exist_ok=True)
        threading.Thread(target=self._writer, daemon=True, name='session-store').start()

    # ---- saving ------------------------------------------------------
    def schedule(self, s):
        """Snapshot the session now (cheap) and write it in the background."""
        keys = s.__dict__.setdefault('_img_keys', {})

        def key(img):
            k = keys.get(id(img))
            if k is None or k[0] is not img:
                k = (img, hashlib.blake2b(np.ascontiguousarray(img).data, digest_size=8).hexdigest())
                keys[id(img)] = k
            return k[1]

        def strip(snap):
            return {'img': key(snap['img']), 'order': snap['order'], 'regions': snap['regions']}

        cur = s.snapshot()
        history = s.history[-HISTORY_KEPT:]
        images = {key(sn['img']): sn['img'] for sn in [cur] + history + s.future}
        s._img_keys = {i: k for i, k in keys.items() if k[1] in images}   # don't keep old images alive
        payload = {'id': s.id, 'exif': s.exif, 'downscaled_from': getattr(s, 'downscaled_from', None),
                   'current': strip(cur), 'history': [strip(sn) for sn in history],
                   'future': [strip(sn) for sn in s.future]}
        with self.cond:
            self.pending[s.id] = (payload, images)
            self.cond.notify()

    def delete(self, sid):
        with self.cond:
            self.pending.pop(sid, None)
        shutil.rmtree(os.path.join(self.root, sid), ignore_errors=True)

    def _writer(self):
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                sid, (payload, images) = self.pending.popitem()
            try:
                self._write(sid, payload, images)
            except Exception as e:   # never take the server down over a save
                self.log(f'could not save session {sid}: {type(e).__name__}: {e}')

    def _write(self, sid, payload, images):
        d = os.path.join(self.root, sid)
        os.makedirs(d, exist_ok=True)
        for k, img in images.items():
            path = os.path.join(d, f'img_{k}.png')
            if not os.path.exists(path):
                tmp = path + '.tmp.png'
                cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                os.replace(tmp, path)
        tmp = os.path.join(d, 'state.tmp')
        with open(tmp, 'wb') as f:
            f.write(zlib.compress(pickle.dumps(payload, protocol=5), 1))
        os.replace(tmp, os.path.join(d, 'state.pkl.z'))
        for path in glob.glob(os.path.join(d, 'img_*.png')):   # images no snapshot refers to any more
            if os.path.basename(path)[4:-4] not in images:
                os.remove(path)

    # ---- loading -----------------------------------------------------
    def load_all(self):
        """The newest `keep` sessions as {sid: Session} (without SAM state); older ones are deleted."""
        dirs = [d for d in glob.glob(os.path.join(self.root, '*')) if os.path.isfile(os.path.join(d, 'state.pkl.z'))]
        dirs.sort(key=lambda d: os.path.getmtime(os.path.join(d, 'state.pkl.z')), reverse=True)
        out = {}
        for d in dirs[self.keep:]:
            shutil.rmtree(d, ignore_errors=True)
        for d in reversed(dirs[:self.keep]):   # oldest first, so dict order = age order
            try:
                s = self._read(d)
                out[s.id] = s
            except Exception as e:
                self.log(f'skipped saved session {os.path.basename(d)}: {type(e).__name__}: {e}')
        return out

    def _read(self, d):
        with open(os.path.join(d, 'state.pkl.z'), 'rb') as f:
            p = pickle.loads(zlib.decompress(f.read()))
        images = {}

        def img(k):
            if k not in images:
                im = cv2.imread(os.path.join(d, f'img_{k}.png'), cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(f'img_{k}.png')
                images[k] = np.ascontiguousarray(im[..., ::-1])
            return images[k]

        def full(snap):
            return {'img': img(snap['img']), 'order': snap['order'], 'regions': snap['regions']}

        cur = full(p['current'])
        s = Session(cur['img'], None)
        s.id = p['id']
        s.exif = p['exif']
        s.downscaled_from = p['downscaled_from']
        s.history = [full(sn) for sn in p['history']]
        s.future = [full(sn) for sn in p['future']]
        s._restore(cur, embed=lambda im: None)
        ids = [int(rid[1:]) for sn in [cur] + s.history + s.future for rid in sn['regions'] if rid[1:].isdigit()]
        reserve_region_ids(max(ids, default=0))
        return s
