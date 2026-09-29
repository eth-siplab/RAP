# --------------------------------------------
# RAP web server
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import hashlib
import io
import json
import os
import threading
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import adjust, geometry
from .models import ModelManager
from .restorers import build_restorers
from .segment import Segmenter
from .subject import DepthModel, SubjectModel
from .session import BACKGROUND, Region, Session, encode_png, stroke_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
EXAMPLES = os.path.join(ROOT, 'examples')
MAX_SIDE = 6000
MAX_EXPORT_SIDE = 8192


class ClickReq(BaseModel):
    region_id: str | None = None
    points: list[list[float]] = []       # [x, y, label] in preview pixels
    box: list[float] | None = None        # [x0, y0, x1, y1] in preview pixels


class PaintReq(BaseModel):
    region_id: str | None = None
    points: list[list[float]]             # stroke polyline in preview pixels
    radius: float                         # brush radius in preview pixels
    add: bool = True


class TransformReq(BaseModel):
    rot: int = 0                          # quarter turns, clockwise
    flip_h: bool = False
    flip_v: bool = False
    angle: float = 0.0                    # straighten, degrees clockwise
    crop: list[float] | None = None       # [x0, y0, x1, y1] normalised in the rotated frame
    fill: bool = True                     # inpaint corners that fall outside the photo


class TextReq(BaseModel):
    prompt: str
    separate: bool = False


class RegionPatch(BaseModel):
    name: str | None = None
    feather: float | None = None
    visible: bool | None = None
    adjust: dict | None = None            # tonal adjustments + 'strength'


class SweepReq(BaseModel):
    restorer: str
    values: dict | None = None


class RenderReq(BaseModel):
    values: dict


class OrderReq(BaseModel):
    order: list[str]


def load_image(data):
    """(RGB array, EXIF bytes or None, original size if it was downscaled). Pixels are
    rotated upright, so Orientation is reset."""
    im = Image.open(io.BytesIO(data))
    exif = im.getexif()
    img = ImageOps.exif_transpose(im).convert('RGB')
    original = None
    if max(img.size) > MAX_SIDE:
        original = list(img.size)
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    exif_bytes = None
    if exif:
        exif[0x0112] = 1
        exif_bytes = exif.tobytes()
    return np.array(img), exif_bytes, original


def create_app(weights_dir, device='cuda', gpu_budget=None, idle_offload=20, preload=('pisasr', 'codeformer'),
               session_dir=None, max_gpu_memory=None):
    log = lambda m: print(m, flush=True)
    models = ModelManager(device, gpu_budget, idle_offload, log, gpu_limit_gb=max_gpu_memory)
    models.register('sam3', 'SAM 3', lambda: Segmenter(device))
    models.get('sam3')                           # every click needs it: load it now

    class SAM:
        """SAM 3 through the model manager, so it moves back to the GPU if it was moved out for room."""
        def __getattr__(self, name):
            return getattr(models.get('sam3'), name)
    seg = SAM()
    models.register('subject', 'BiRefNet · Subject', lambda: SubjectModel(device))
    models.register('depth', 'Depth Anything V2', lambda: DepthModel(device))
    restorers = build_restorers(weights_dir, device, models, log)
    subject = lambda img: models.get('subject')(img)
    gpu = threading.RLock()  # one GPU job at a time; endpoints run in a thread pool
    sessions: dict[str, Session] = {}
    store = None
    if session_dir:
        from .store import SessionStore
        store = SessionStore(session_dir, keep=4, log=log)
        sessions.update(store.load_all())
        if sessions:
            log(f'restored {len(sessions)} session(s) from {session_dir}')
    app = FastAPI(title='RAP · Restore Anything Pipeline')
    models.preload([k for k in preload if k in models.entries])
    models.start_idle_offload(gpu)

    @app.exception_handler(RuntimeError)
    async def model_error(request, exc):
        # e.g. a model that failed to load on first use
        return JSONResponse({'detail': str(exc)}, status_code=503)

    @app.middleware('http')
    async def history_header(request, call_next):
        # Lets the UI keep its undo/redo buttons accurate without extra requests.
        response = await call_next(request)
        parts = request.url.path.split('/')
        if parts[1] != 'api':
            # Page files: always revalidate, so an updated app.js never runs against a stale index.html.
            response.headers['Cache-Control'] = 'no-cache'
        if len(parts) > 3 and parts[1] == 'api' and parts[2] == 'session' and parts[3] in sessions:
            s = sessions[parts[3]]
            response.headers['X-History'] = f'{int(bool(s.history))},{int(bool(s.future))}'
            response.headers['X-Base'] = base_tag(s)   # the page refetches the edited photo when this changes
            if store and request.method != 'GET' and response.status_code < 400:
                try:
                    store.schedule(s)
                except Exception as e:   # e.g. a concurrent edit; the next change saves again
                    log(f'session save skipped: {e}')
        return response

    # ---- helpers -----------------------------------------------------
    def get(sid):
        if sid not in sessions:
            raise HTTPException(404, 'session expired, please reload the image')
        s = sessions[sid]
        s.last_used = time.time()
        return s

    def drop_caches():
        """Release recomputable GPU state when memory runs out: every region's prepared crops,
        and the SAM features of all but the most recently used image."""
        recent = max(sessions.values(), key=lambda s: getattr(s, 'last_used', 0), default=None)
        for s in list(sessions.values()):
            for r in list(s.regions.values()):
                r.cache = {}
            if s is not recent:
                s.seg_state = None
    models.cache_droppers.append(drop_caches)

    def seg_of(s):
        """The SAM image state; sessions restored from disk compute it on first use. Call with `gpu` held."""
        if s.seg_state is None:
            s.seg_state = seg.set_image(s.img)
        return s.seg_state

    def region_of(s, rid):
        if rid not in s.regions:
            raise HTTPException(404, 'unknown region')
        return s.regions[rid]

    def rbox(s, r, key=None):
        """Crop box of a region for a restorer (its context margin differs per restorer)."""
        return r.crop_box(s.W, s.H, restorers[key or r.restorer].context)

    def region_info(s, r):
        box = rbox(s, r)
        info = {'id': r.id, 'name': r.name, 'source': r.source, 'restorer': r.restorer,
                'blend': restorers[r.restorer].blend,
                'values': r.values, 'adjust': r.adjust, 'feather': r.feather * s.scale, 'visible': r.visible,
                'score': r.score, 'box': s.box_to_preview(box) if box else None,
                'points': [[x * s.scale, y * s.scale, l] for x, y, l in r.points],
                'mask_url': None, 'overlay_url': None}
        if r.mask is not None and box is not None:
            old = getattr(r, '_mask_urls', [])
            s.drop_blobs(old)
            mask, _ = s.mask_png(r)
            info['mask_url'] = s.put_blob(encode_png(mask))
            info['overlay_url'] = s.put_blob(encode_png(s.outline_png(r)))
            r._mask_urls = [info['mask_url'], info['overlay_url']]
        return info

    def session_info(s):
        pw, ph = s.preview_size()
        return {'id': s.id, 'width': s.W, 'height': s.H, 'preview': [pw, ph], 'scale': s.scale,
                'can_undo': bool(s.history), 'can_redo': bool(s.future),
                'downscaled_from': getattr(s, 'downscaled_from', None),
                'image_url': s.put_blob(encode_png(s.preview(s.img))),
                'base_url': base_url(s), 'base_tag': base_tag(s),
                'regions': [region_info(s, s.regions[rid]) for rid in s.order]}

    def new_session(img, exif=None, downscaled_from=None):
        with gpu:
            state = seg.set_image(img)
        s = Session(img, state)
        s.exif = exif
        s.downscaled_from = downscaled_from
        # Keep memory bounded: this is a single-user tool, a few images is plenty.
        while len(sessions) >= 4:
            old = sessions.pop(next(iter(sessions)))
            if store:
                store.delete(old.id)
        sessions[s.id] = s
        if store:
            store.schedule(s)
        return session_info(s)

    # ---- edits: removals and replacements change the photo the other layers work on
    def is_edit(s, r):
        return (r.id != BACKGROUND and r.visible and r.restorer != 'none'
                and getattr(restorers[r.restorer], 'edits_photo', False))

    def mask_tag(r):
        """Short hash of a region's mask (masks are replaced, never changed in place)."""
        if r.mask is None:
            return None
        t = r.__dict__.get('_mask_tag')
        if t is None or t[0] is not r.mask:
            t = (r.mask, hashlib.md5(np.packbits(r.mask).tobytes()).hexdigest()[:12])
            r._mask_tag = t
        return t[1]

    def base_key(s):
        return tuple((r.id, r.restorer, json.dumps(r.values, sort_keys=True), mask_tag(r),
                      json.dumps(r.adjust, sort_keys=True))
                     for r in (s.regions[rid] for rid in s.order) if is_edit(s, r))

    def base_tag(s):
        k = base_key(s)
        return 'orig' if not k else hashlib.md5(repr(k).encode()).hexdigest()[:10]

    def changed_alpha(orig, out, grow=4):
        """Where an edit changed its crop, as a soft 0..1 mask (edits leave the rest untouched)."""
        d = np.abs(out.astype(np.int16) - orig.astype(np.int16)).max(-1) > 3
        m = cv2.dilate(d.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
        return np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), grow / 2), d)

    def edit_output(s, r, box):
        """Full-resolution result of an edit region at its current settings (kept per region)."""
        res = restorers[r.restorer]
        values = {**res.defaults(), **r.values}
        k = (cache_key(s, r, res.key, box), tuple(sorted(values.items())))
        full = r.__dict__.setdefault('full', {})
        if k not in full:
            full[k] = res.render(prepared(s, r, res.key, box), values)
        return full[k]

    def base(s):
        """The photo with every removal/replacement applied, bottom to top; the background and
        the other regions are processed on this. Call with `gpu` held (it may render)."""
        key = base_key(s)
        cached = s.__dict__.get('_base')
        if cached is not None and cached[0] == key and cached[1] is s.img:
            return cached[2]
        img = s.img
        if key:
            img = s.img.copy()
            for rid in s.order:
                r = s.regions[rid]
                box = rbox(s, r) if is_edit(s, r) else None
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                orig = s.img[y0:y1, x0:x1]
                out = edit_output(s, r, box)
                if not adjust.is_identity(r.adjust, True):
                    out = adjust.apply(orig, out, r.adjust, s.scale, (x0, y0), (s.W, s.H))
                a = changed_alpha(orig, out)[..., None]
                cur = img[y0:y1, x0:x1].astype(np.float32)
                img[y0:y1, x0:x1] = (cur * (1 - a) + out.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)
        s._base = (key, s.img, img)
        return img

    def base_url(s):
        """Preview of the edited photo (the plain image when there are no edits)."""
        with gpu:
            b = base(s)
        if b is s.img:
            return None
        tag = base_tag(s)
        old = s.__dict__.get('_base_url')
        if old and old[0] == tag and old[1] in s.blobs:
            return old[1]
        url = s.put_blob(encode_png(s.preview(b)))
        if old:
            s.drop_blobs([old[1]])
        s._base_url = (tag, url)
        return url

    def foreground(s, exclude):
        """Union of the other visible region masks (what a background effect keeps sharp).
        Edited regions are not foreground: after a removal that area belongs to the background."""
        m = None
        for rid in s.order:
            o = s.regions[rid]
            if rid in (BACKGROUND, exclude) or not o.visible or o.mask is None or is_edit(s, o):
                continue
            m = o.mask.copy() if m is None else (m | o.mask)
        return m

    def cache_key(s, r, key, box):
        """(restorer, crop, foreground, source photo, mask): prepared work is reused while these match."""
        res = restorers[key]
        edit = getattr(res, 'edits_photo', False) and r.id != BACKGROUND
        src = 'orig' if edit else base_tag(s)
        mask = mask_tag(r) if res.needs_mask else None
        fg = None
        if res.uses_foreground:
            m = foreground(s, r.id)
            fg = 'none' if m is None else hashlib.md5(np.packbits(m).tobytes()).hexdigest()[:12]
        return (key, box, fg, src, mask)

    def prepared(s, r, key, box):
        ck = cache_key(s, r, key, box)
        if ck not in r.cache:
            # The crop (or the foreground it depends on) changed: forget stale work.
            r.cache = {k: v for k, v in r.cache.items() if k[1] == box and (k[0] != key or k[2:] == ck[2:])}
            stale = [k for k in r.renders if k[0][1] != box or (k[0][0] == key and k[0][2:] != ck[2:])]
            s.drop_blobs([r.renders.pop(k) for k in stale])
            x0, y0, x1, y1 = box
            res = restorers[key]
            src = s.img if ck[3] == 'orig' else base(s)   # edits work on the photo, the rest on the edited photo
            crop = np.ascontiguousarray(src[y0:y1, x0:x1])
            mask = None
            if res.needs_mask and r.mask is not None:
                mask = np.ascontiguousarray(r.mask[y0:y1, x0:x1])
            elif res.needs_mask:   # background layer: everything outside the other regions
                fg = foreground(s, r.id)
                mask = np.ones(crop.shape[:2], bool) if fg is None else ~np.ascontiguousarray(fg[y0:y1, x0:x1])
            elif res.uses_foreground:
                fg = foreground(s, r.id)
                mask = None if fg is None else np.ascontiguousarray(fg[y0:y1, x0:x1])
            r.cache[ck] = res.prepare(crop, mask)
        return r.cache[ck]

    def render_urls(s, r, key, box, values_list):
        """Blob urls for several settings, rendering the uncached ones in one batch."""
        state = prepared(s, r, key, box)
        ck = cache_key(s, r, key, box)
        keys = [(ck, tuple(sorted(v.items()))) for v in values_list]
        todo = [(k, v) for k, v in zip(keys, values_list) if k not in r.renders]
        todo = list({k: v for k, v in todo}.items())
        if todo:
            outs = restorers[key].render_many(state, [v for _, v in todo])
            edit = getattr(restorers[key], 'edits_photo', False) and r.id != BACKGROUND
            full = r.__dict__.setdefault('full', {})
            for (k, _), out in zip(todo, outs):
                r.renders[k] = s.put_blob(encode_png(s.preview(out)))
                if edit:   # full-resolution results, so the edited photo needs no second render
                    full[k] = out
            if len(full) > 12:
                for k in list(full)[:-12]:
                    del full[k]
        return [r.renders[k] for k in keys]

    # ---- routes ------------------------------------------------------
    def examples():
        exts = ('.png', '.jpg', '.jpeg', '.webp')
        files = sorted(f for f in os.listdir(EXAMPLES) if f.lower().endswith(exts)) if os.path.isdir(EXAMPLES) else []
        meta = {}
        path = os.path.join(EXAMPLES, 'examples.json')
        if os.path.isfile(path):
            import json
            meta = {m['file']: m for m in json.load(open(path))}
        order = [f for f in meta if f in files] + [f for f in files if f not in meta]
        return [{'file': f, 'title': meta.get(f, {}).get('title', f), 'hint': meta.get(f, {}).get('hint', ''),
                 'credit': meta.get(f, {}).get('credit', ''), 'source': meta.get(f, {}).get('source', '')}
                for f in order]

    @app.get('/api/models')
    def model_status():
        return models.status()

    @app.post('/api/models/free')
    def free_models():
        models.offload_all(gpu)
        return models.status()

    @app.get('/api/config')
    def config():
        return {'restorers': [r.spec() for r in restorers.values()], 'adjustments': adjust.SPEC,
                'examples': examples()}

    thumbs = {}

    @app.get('/api/example/{name}')
    def example_file(name: str, thumb: int = 0):
        path = os.path.join(EXAMPLES, os.path.basename(name))
        if not os.path.isfile(path):
            raise HTTPException(404)
        if not thumb:
            return FileResponse(path)
        if name not in thumbs:
            im = Image.open(path).convert('RGB')
            im.thumbnail((480, 480), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=85)
            thumbs[name] = buf.getvalue()
        return Response(thumbs[name], media_type='image/jpeg', headers={'Cache-Control': 'max-age=86400'})

    @app.post('/api/session')
    async def upload(file: UploadFile = File(...)):
        try:
            img, exif, original = load_image(await file.read())
        except Exception as e:
            raise HTTPException(400, f'could not read image: {e}')
        return new_session(img, exif, original)

    @app.post('/api/session/example/{name}')
    def open_example(name: str):
        path = os.path.join(EXAMPLES, os.path.basename(name))
        if not os.path.isfile(path):
            raise HTTPException(404)
        with open(path, 'rb') as f:
            return new_session(*load_image(f.read()))

    @app.get('/api/session/{sid}/base')
    def edited_photo(sid: str):
        """The photo with the removals/replacements applied, which the other layers work on."""
        s = get(sid)
        return {'url': base_url(s), 'tag': base_tag(s)}

    @app.get('/api/session/{sid}')
    def get_session(sid: str):
        """Re-attach to a session (e.g. after reloading the page)."""
        return session_info(get(sid))

    @app.post('/api/session/{sid}/analyze')
    def analyze(sid: str):
        """Cheap look at the photo to suggest what to do first."""
        s = get(sid)
        img = s.img
        small = cv2.resize(img, (max(1, round(img.shape[1] * 512 / max(img.shape[:2]))),
                                 max(1, round(img.shape[0] * 512 / max(img.shape[:2])))), interpolation=cv2.INTER_AREA)
        f = small.astype(np.float32) / 255
        luma = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        out = []
        if np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean() < 0.02 and 'ddcolor' in restorers:
            out.append({'id': 'colorize', 'title': 'Black-and-white photo', 'detail': 'Add colour with DDColor.',
                        'action': {'type': 'sweep', 'region': 'bg', 'restorer': 'ddcolor'}})
        if np.median(luma) < 0.18 and 'cidnet' in restorers:
            out.append({'id': 'lowlight', 'title': 'Looks under-exposed', 'detail': 'Brighten it with HVI-CIDNet.',
                        'action': {'type': 'sweep', 'region': 'bg', 'restorer': 'cidnet'}})
        if 'fbcnn_deblock' in restorers:
            with gpu:
                qf = restorers['fbcnn_deblock'].estimate(restorers['fbcnn_deblock'].prepare(
                    np.ascontiguousarray(img[:1024, :1024]))).get('qf')
            if qf is not None and qf < 50:
                out.append({'id': 'jpeg', 'title': f'Strong JPEG compression (quality ≈ {qf:.0f})',
                            'detail': 'Remove the blocks and ringing with FBCNN.',
                            'action': {'type': 'sweep', 'region': 'bg', 'restorer': 'fbcnn_deblock'}})
        with gpu:
            faces = seg.from_text(seg_of(s), 'human face')
        faces = [m for m, sc in faces if sc > 0.6]
        if faces and 'codeformer' in restorers:
            n = len(faces)
            out.append({'id': 'faces', 'title': f'{n} face{"s" if n > 1 else ""} found',
                        'detail': 'Restore them with CodeFormer (and the rest with PiSA-SR).', 'action': {'type': 'auto'}})
        # noise: Immerkaer's estimator on a full-resolution centre crop
        c = img[max(0, img.shape[0] // 2 - 512):img.shape[0] // 2 + 512, max(0, img.shape[1] // 2 - 512):img.shape[1] // 2 + 512]
        L = (c.astype(np.float32) / 255) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
        sigma = np.sqrt(np.pi / 2) * np.abs(cv2.filter2D(L, -1, k))[1:-1, 1:-1].mean() / 6
        sharp = cv2.Laplacian(luma, cv2.CV_32F).var()
        seen = {o['id'] for o in out}
        denoiser = next((k for k in ('scunet', 'pisasr') if k in restorers), None)
        if sigma > 0.065 and denoiser and 'jpeg' not in seen:
            out.append({'id': 'noise', 'title': f'Visible noise (σ ≈ {sigma * 255:.0f})',
                        'detail': f'Clean it up with {"SCUNet" if denoiser == "scunet" else "PiSA-SR"}.',
                        'action': {'type': 'sweep', 'region': 'bg', 'restorer': denoiser}})
        elif sharp < 2e-3 and 'pisasr' in restorers and 'jpeg' not in seen:
            out.append({'id': 'soft', 'title': 'Looks soft or blurry', 'detail': 'Sharpen it with PiSA-SR.',
                        'action': {'type': 'sweep', 'region': 'bg', 'restorer': 'pisasr'}})
        if max(s.W, s.H) < 1200:
            out.append({'id': 'small', 'title': f'Small image ({s.W} × {s.H})',
                        'detail': 'Export at 2× or 4× to upscale it with PiSA-SR.', 'action': {'type': 'export', 'size': '4' if max(s.W, s.H) < 700 else '2'}})
        if not out and 'lens_blur' in restorers:
            out.append({'id': 'portrait', 'title': 'Looks clean', 'detail': 'Try portrait mode: keep the subject sharp, blur the background.',
                        'action': {'type': 'portrait'}})
        return {'suggestions': out[:4]}

    @app.get('/api/blob/{sid}/{token}')
    def blob(sid: str, token: str):
        s = get(sid)
        if token not in s.blobs:
            raise HTTPException(404)
        data, mime = s.blobs[token]
        return Response(data, media_type=mime, headers={'Cache-Control': 'private, max-age=3600'})

    @app.post('/api/session/{sid}/click')
    def click(sid: str, req: ClickReq):
        s = get(sid)
        if req.region_id and req.region_id != BACKGROUND:
            r = region_of(s, req.region_id)
        else:
            r = None
        points = [(s.to_full(x), s.to_full(y), int(l)) for x, y, l in req.points]
        box = [s.to_full(v) for v in req.box] if req.box else None
        if not points and box is None:
            raise HTTPException(400, 'no prompt')
        # Re-run from all prompts; reuse the previous mask only when prompts were added.
        prev = r.logits if (r is not None and r.source == 'click' and len(points) > len(r.points)
                            and r.box == box) else None
        with gpu:
            mask, score, logits = seg.from_prompts(seg_of(s), points, box, prev)
        if not mask.any():
            raise HTTPException(422, 'nothing found at that location')
        s.push()
        if r is None:
            r = s.add_region(Region(f'Region {len(s.order)}', mask, 'click'))
        else:
            r.mask, r.source, r.soft = r.apply_strokes(mask), 'click', None
        r.points, r.box, r.logits, r.score = points, box, logits, score
        return region_info(s, r)

    @app.post('/api/session/{sid}/paint')
    def paint(sid: str, req: PaintReq):
        """Brush stroke: add to or erase from a region's mask (a new region if none given)."""
        s = get(sid)
        pts = [(s.to_full(x), s.to_full(y)) for x, y in req.points]
        radius = s.to_full(req.radius)
        if not pts:
            raise HTTPException(400, 'empty stroke')
        if req.region_id and req.region_id != BACKGROUND:
            r = region_of(s, req.region_id)
            s.push()
        else:
            if not req.add:
                raise HTTPException(400, 'nothing to erase from')
            s.push()
            r = s.add_region(Region(f'Region {len(s.order)}', np.zeros((s.H, s.W), bool), 'brush'))
        r.strokes.append((pts, radius, req.add))
        stroke = stroke_mask((s.H, s.W), pts, radius)
        r.mask = (r.mask | stroke) if req.add else (r.mask & ~stroke)
        if r.soft is not None:   # keep the matte's soft edges away from the stroke
            r.soft = np.where(stroke, 255 if req.add else 0, r.soft).astype(np.uint8)
        if not r.mask.any():
            s.undo(lambda img: s.seg_state)
            s.future.clear()
            raise HTTPException(422, 'the region would be empty')
        return region_info(s, r)

    def embedder():
        def embed(img):
            with gpu:
                return seg.set_image(img)
        return embed

    @app.post('/api/session/{sid}/undo')
    def undo(sid: str):
        s = get(sid)
        if not s.undo(embedder()):
            raise HTTPException(409, 'nothing to undo')
        return session_info(s)

    @app.post('/api/session/{sid}/redo')
    def redo(sid: str):
        s = get(sid)
        if not s.redo(embedder()):
            raise HTTPException(409, 'nothing to redo')
        return session_info(s)

    @app.post('/api/session/{sid}/text')
    def text(sid: str, req: TextReq):
        s = get(sid)
        prompt = req.prompt.strip()
        if not prompt:
            raise HTTPException(400, 'empty prompt')
        with gpu:
            found = seg.from_text(seg_of(s), prompt)
        if not found:
            raise HTTPException(422, f'no "{prompt}" found in the image')
        s.push()
        if req.separate:
            new = [Region(f'{prompt} {i + 1}', m, 'text') for i, (m, _) in enumerate(found)]
            for r, (_, sc) in zip(new, found):
                r.score = sc
        else:
            r = Region(prompt, np.any([m for m, _ in found], axis=0), 'text')
            r.score = max(sc for _, sc in found)
            new = [r]
        for r in new:
            s.add_region(r)
        return {'regions': [region_info(s, r) for r in new], 'count': len(found)}

    @app.post('/api/session/{sid}/subject')
    def select_subject(sid: str):
        """Matte the main subject (BiRefNet) into a new soft-edged region."""
        s = get(sid)
        with gpu:
            alpha = subject(s.img)
            torch.cuda.empty_cache()
        mask = alpha >= 128
        if mask.sum() < 64:
            raise HTTPException(422, 'no clear subject found')
        s.push()
        r = s.add_region(Region('Subject', mask, 'subject'))
        r.soft = alpha
        return region_info(s, r)

    @app.patch('/api/session/{sid}/region/{rid}')
    def patch_region(sid: str, rid: str, req: RegionPatch):
        s = get(sid)
        r = region_of(s, rid)
        s.push()
        if req.name is not None:
            r.name = req.name[:80]
        if req.visible is not None:
            r.visible = req.visible
        if req.feather is not None:
            r.feather = max(0.0, float(req.feather)) / s.scale
        if req.adjust is not None:
            ranges = {a['key']: (a['min'], a['max']) for a in adjust.SPEC}
            ranges['strength'] = (0.0, 1.0)
            r.adjust = {k: float(np.clip(float(v), *ranges[k])) for k, v in req.adjust.items() if k in ranges}
            if req.name is None and req.feather is None and req.visible is None:
                return {'id': r.id, 'adjust': r.adjust}   # no need to re-encode the mask
        return region_info(s, r)

    @app.delete('/api/session/{sid}/region/{rid}')
    def delete_region(sid: str, rid: str):
        s = get(sid)
        r = region_of(s, rid)
        s.push()
        s.drop_blobs(getattr(r, '_mask_urls', []) + list(r.renders.values()))
        s.remove_region(rid)
        return {'order': s.order}

    @app.post('/api/session/{sid}/order')
    def reorder(sid: str, req: OrderReq):
        s = get(sid)
        rest = [rid for rid in req.order if rid in s.regions and rid != BACKGROUND]
        if sorted(rest) != sorted(r for r in s.order if r != BACKGROUND):
            raise HTTPException(400, 'order must list every region')
        s.push()
        s.order = [BACKGROUND] + rest
        return {'order': s.order}

    @app.post('/api/session/{sid}/region/{rid}/sweep')
    def sweep(sid: str, rid: str, req: SweepReq):
        """Assign a restorer and precompute its primary-control sweep."""
        s = get(sid)
        r = region_of(s, rid)
        if req.restorer not in restorers:
            raise HTTPException(400, f'unknown restorer {req.restorer}')
        res = restorers[req.restorer]
        if res.needs_mask and r.mask is None and not res.allow_bg:
            raise HTTPException(400, f'{res.label} needs a selected region, not the background')
        if res.bg_only and r.mask is not None:
            raise HTTPException(400, f'{res.label} is only available for the background')
        box = rbox(s, r, res.key)
        if box is None:
            raise HTTPException(422, 'region is empty')
        before = (r.restorer, dict(r.values))
        snap = s.snapshot()
        out = {'restorer': res.key, 'box': s.box_to_preview(box), 'blend': res.blend, 'candidates': [],
               'estimate': {}, 'info': ''}
        if res.key == 'none':
            if before[0] != 'none':
                s.history.append(snap); s.future.clear()
            r.restorer, r.values = 'none', {}
            out['values'] = {}
            return out
        with gpu:
            state = prepared(s, r, res.key, box)
            est = res.estimate(state)
            values = {**res.defaults(), **est, **(req.values or {})}
            if res.blend == 'transparent':
                out['current'] = None
            elif not res.params:
                out['current'] = render_urls(s, r, res.key, box, [values])[0]
            else:
                primary = res.params[0]
                settings = [{**values, primary.key: v} for v in primary.sweep] + [values]
                urls = render_urls(s, r, res.key, box, settings)
                out['candidates'] = [{'value': v, 'url': u} for v, u in zip(primary.sweep, urls)]
                out['current'] = urls[-1]
        torch.cuda.empty_cache()  # be a good neighbour on a shared GPU
        if hasattr(res, 'info'):
            out['info'] = res.info(state)
        if before != (res.key, values):
            s.history.append(snap); s.future.clear()
        r.restorer, r.values = res.key, values
        out['values'] = values
        out['estimate'] = est
        return out

    @app.post('/api/session/{sid}/region/{rid}/render')
    def render(sid: str, rid: str, req: RenderReq):
        s = get(sid)
        r = region_of(s, rid)
        if r.restorer == 'none':
            raise HTTPException(400, 'no restorer assigned')
        res = restorers[r.restorer]
        box = rbox(s, r)
        values = {**res.defaults(), **{k: v if isinstance(v, str) else float(v) for k, v in req.values.items()}}
        with gpu:
            url = render_urls(s, r, res.key, box, [values])[0]
        if values != r.values:
            s.push()
        r.values = values
        return {'url': url, 'box': s.box_to_preview(box)}

    def is_transparent(s):
        bg = s.regions[BACKGROUND]
        return bg.visible and restorers[bg.restorer].blend == 'transparent'

    def full_composite(s):
        """The result at full resolution: RGB, or RGBA when the background is transparent."""
        renders = {}
        transparent = is_transparent(s)
        with gpu:
            b = base(s)
            for rid in s.order:
                r = s.regions[rid]
                if (transparent and rid == BACKGROUND) or is_edit(s, r):   # edits are in the base
                    continue
                has_restorer = r.restorer != 'none'
                if not r.visible or (not has_restorer and adjust.is_identity(r.adjust, False) and not transparent):
                    continue
                box = rbox(s, r)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                orig = b[y0:y1, x0:x1]
                res = restorers[r.restorer]
                out = None
                if has_restorer:
                    out = res.render(prepared(s, r, res.key, box), {**res.defaults(), **r.values})
                if not adjust.is_identity(r.adjust, has_restorer):
                    out = adjust.apply(orig, out, r.adjust, s.scale, (x0, y0), (s.W, s.H))
                if out is None:
                    out = orig
                alpha = np.ones(out.shape[:2], np.float32) if res.blend == 'box' else None
                renders[rid] = (out, box, alpha)
            torch.cuda.empty_cache()
        return s.composite(renders, transparent, base=b)

    def rebase_with(s, img, under):
        """Make a composite the new base. An RGBA result keeps its cut-out as a soft
        region over a transparent background; `under` fills the colour behind it."""
        cutout = None
        if img.shape[2] == 4:
            a = img[..., 3:].astype(np.float32) / 255
            img, cutout = (img[..., :3] * a + under * (1 - a)).round().astype(np.uint8), img[..., 3]
        with gpu:
            state = seg.set_image(img)
        s.push()
        s.rebase(img, state)
        if cutout is not None and cutout.max() > 0:
            r = s.add_region(Region('Cut-out', cutout >= 128, 'subject'))
            r.soft = cutout
            s.regions[BACKGROUND].restorer = 'transparent'

    @app.get('/api/session/{sid}/export')
    def export(sid: str, scale: float = 1.0, fmt: str = 'png', quality: int = 92, max_side: int = 0,
               meta: bool = True):
        """Full-resolution result. scale > 1 super-resolves with PiSA-SR, max_side downsizes."""
        s = get(sid)
        scale = max(1.0, min(float(scale), 4.0, MAX_EXPORT_SIDE / max(s.H, s.W)))
        out = full_composite(s)
        if scale > 1.01 and 'pisasr' in restorers:
            with gpu:
                rgb = restorers['pisasr'].upscale(np.ascontiguousarray(out[..., :3]), scale)
                torch.cuda.empty_cache()
            if out.shape[2] == 4:   # alpha is resized, not hallucinated
                a = cv2.resize(out[..., 3], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
                rgb = np.concatenate([rgb, a[..., None]], -1)
            out = rgb
        im = Image.fromarray(out)
        if im.mode == 'RGBA' and fmt.lower() in ('jpeg', 'jpg'):   # no alpha in JPEG: flatten on white
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        fmt = fmt.lower()
        ext = {'png': 'png', 'jpeg': 'jpg', 'jpg': 'jpg', 'webp': 'webp'}.get(fmt)
        if ext is None:
            raise HTTPException(400, f'unsupported format {fmt}')
        kw = {'exif': s.exif} if meta and s.exif else {}
        buf = io.BytesIO()
        if ext == 'png':
            im.save(buf, format='PNG', compress_level=6, **kw)
            mime = 'image/png'
        elif ext == 'jpg':
            q = max(1, min(int(quality), 100))
            im.save(buf, format='JPEG', quality=q, subsampling=0 if q >= 90 else 2, optimize=True, **kw)
            mime = 'image/jpeg'
        else:
            im.save(buf, format='WEBP', quality=max(1, min(int(quality), 100)), method=4, **kw)
            mime = 'image/webp'
        name = f'restored{f"_x{scale:g}" if scale > 1.01 else ""}.{ext}'
        return Response(buf.getvalue(), media_type=mime,
                        headers={'Content-Disposition': f'attachment; filename="{name}"',
                                 'X-Export-Scale': f'{scale:g}', 'X-Export-Name': name,
                                 'X-Export-Size': f'{im.width}x{im.height}'})

    @app.post('/api/session/{sid}/apply')
    def apply_all(sid: str):
        """Bake the current result into a new base image and start over with it."""
        s = get(sid)
        rebase_with(s, full_composite(s), s.img)
        return session_info(s)

    @app.post('/api/session/{sid}/transform')
    def transform(sid: str, req: TransformReq):
        """Crop / rotate / flip the current result; it becomes the new base image."""
        s = get(sid)
        img = full_composite(s)
        args = (req.rot, req.flip_h, req.flip_v, req.angle, req.crop)
        img, invalid = geometry.transform(img, *args)
        under, _ = geometry.transform(s.img, *args)
        if min(img.shape[:2]) < 8:
            raise HTTPException(400, 'crop is too small')
        if invalid is not None and req.fill and 'lama' in restorers:
            hole = cv2.dilate(invalid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            with gpu:
                lama = restorers['lama']
                rgb = lama.render(lama.prepare(np.ascontiguousarray(img[..., :3]), hole), {'expand': 0})
                under = lama.render(lama.prepare(under, hole), {'expand': 0}) if img.shape[2] == 4 else rgb
            img = np.concatenate([rgb, img[..., 3:]], -1) if img.shape[2] == 4 else rgb
        rebase_with(s, img, under)
        return session_info(s)

    @app.post('/api/client-error')
    async def client_error(request: Request):
        # Script errors from the page end up in the server log.
        body = (await request.body())[:4000].decode('utf-8', 'replace')
        print('client error:', body, flush=True)
        return {'ok': True}

    app.mount('/', StaticFiles(directory=STATIC, html=True), name='static')
    return app
