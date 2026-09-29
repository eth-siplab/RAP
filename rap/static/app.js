// --------------------------------------------
// RAP web interface
// --------------------------------------------
// Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
// https://github.com/eth-siplab/RAP
// Jiaxi Jiang (https://jiaxi-jiang.com/)
// Sensing, Interaction & Perception Lab,
// Department of Computer Science, ETH Zurich

'use strict';

// Region colours (for swatches, outlines and click markers).
const COLORS = ['#5b9dff', '#ff8a5b', '#5bd49a', '#e36bd6', '#f2c94c', '#56ccf2', '#ff6b8b', '#a78bfa'];

const S = {
  restorers: {},       // key -> spec
  adjSpec: [],         // tonal adjustments (from /api/config)
  order: [],           // restorer keys in UI order
  sess: null,
  orig: null,          // HTMLImageElement of the preview-size original
  regions: [],         // bottom -> top; regions[0] is the background
  active: null,        // id of the highlighted region
  editing: null,       // id of the region that receives clicks (null: a click starts a new one)
  tool: 'point',
  view: { x: 0, y: 0, k: 1 },
  compare: false,
  split: null,         // null or 0..1
  drag: null,
  busy: 0,
  brush: 24,           // brush radius in screen pixels
  hover: null,         // last pointer position (screen) for the brush cursor
  crop: null,          // crop-mode state, see enterCrop()
};

const $ = (s) => document.querySelector(s);
const canvas = $('#canvas'), ctx = canvas.getContext('2d');
const overlay = $('#overlay'), octx = overlay.getContext('2d');
const viewport = $('#viewport'), stage = $('#stage');

// ------------------------------------------------------------------ utils
function toast(msg, ok = false, ms = 0) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (ok ? ' ok' : '');
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms || (ok ? 2000 : 4500));
}

async function busy(label, fn) {
  if (S.busy++ === 0) {
    const t0 = Date.now();
    $('#busy-time').textContent = '';
    busy.timer = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      $('#busy-time').textContent = s >= 2 ? `${s} s` : '';
    }, 500);
  }
  $('#busy').hidden = false;
  $('#busy-text').textContent = label;
  try { return await fn(); }
  catch (e) { toast(e.message || String(e)); throw e; }
  finally {
    if (--S.busy === 0) { $('#busy').hidden = true; clearInterval(busy.timer); }
  }
}

async function api(path, body, method) {
  const opt = { method: method || (body !== undefined ? 'POST' : 'GET') };
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) { opt.body = JSON.stringify(body); opt.headers = { 'Content-Type': 'application/json' }; }
  const r = await fetch(path, opt);
  const h = r.headers.get('X-History');
  if (h) {
    const [u, d] = h.split(',');
    $('#btn-undo-all').disabled = u !== '1'; $('#btn-redo-all').disabled = d !== '1';
    if (opt.method !== 'GET' && u === '1') S.dirty = true;   // unsaved changes (for the leave warning)
  }
  const b = r.headers.get('X-Base');
  if (b && S.sess && b !== S.baseTag && path.startsWith(`/api/session/${S.sess.id}`)) {
    S.baseTag = b;
    setTimeout(syncBase, 0);
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function loadImg(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('failed to load ' + url));
    im.src = url;
  });
}

const sid = () => S.sess.id;
const region = (id) => S.regions.find((r) => r.id === id);
const colorOf = (r) => r.id === 'bg' ? '#888' : COLORS[(parseInt(r.id.slice(1)) - 1) % COLORS.length];
const fmt = (v, step) => (step >= 1 ? String(Math.round(v)) : Number(v).toFixed(step >= 0.1 ? 1 : 2));

// ------------------------------------------------------------ adjustments
// Mirrors rap/adjust.py, which is what export uses; keep the two in sync.
const WB_STOPS = 0.3, CURVE_NORM = 9.4815, CLARITY_R = 0.01, DEHAZE_R = 0.03;
const srgbToLin = (x) => (x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4));
const linToSrgb = (y) => { y = Math.max(y, 0); return y <= 0.0031308 ? 12.92 * y : 1.055 * Math.pow(y, 1 / 2.4) - 0.055; };
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

function adjValue(r, key) {
  if (r.adjust && r.adjust[key] !== undefined) return r.adjust[key];
  if (key === 'strength') return 1;
  const a = S.adjSpec.find((x) => x.key === key);
  return a ? a.default : 0;
}

function adjIdentity(r) {
  if (r.restorer !== 'none' && adjValue(r, 'strength') !== 1) return false;
  return S.adjSpec.every((a) => adjValue(r, a.key) === a.default);
}

function toneLuts(r) {
  const g = (k) => adjValue(r, k);
  const ev = g('exposure'), c = g('contrast'), t = g('temperature'), ti = g('tint');
  const hl = g('highlights'), sh = g('shadows'), lo = -0.1 * g('blacks'), hi = 1 - 0.1 * g('whites');
  const gains = [Math.pow(2, WB_STOPS * t), Math.pow(2, -WB_STOPS * ti), Math.pow(2, -WB_STOPS * t)];
  const lut = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const lin = srgbToLin(i / 255);
    for (let ch = 0; ch < 3; ch++) {
      let v = (linToSrgb(lin * Math.pow(2, ev) * gains[ch]) - lo) / (hi - lo);
      const vc = clamp01(v);
      v += 0.2 * CURVE_NORM * (sh * vc * (1 - vc) ** 3 + hl * vc ** 3 * (1 - vc));
      lut[i * 3 + ch] = (v - 0.5) * (1 + c) + 0.5;
    }
  }
  return lut;
}

// Three box blurs of radius r with replicated borders (cv2.blur x3 in adjust.py).
function box3(src, w, h, r) {
  if (r < 1) return src;
  let a = Float32Array.from(src), b = new Float32Array(w * h);
  const n = 2 * r + 1;
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) {           // horizontal
      const row = y * w;
      let acc = 0;
      for (let k = -r; k <= r; k++) acc += a[row + Math.min(Math.max(k, 0), w - 1)];
      for (let x = 0; x < w; x++) {
        b[row + x] = acc / n;
        acc += a[row + Math.min(x + r + 1, w - 1)] - a[row + Math.max(x - r, 0)];
      }
    }
    for (let x = 0; x < w; x++) {           // vertical
      let acc = 0;
      for (let k = -r; k <= r; k++) acc += b[Math.min(Math.max(k, 0), h - 1) * w + x];
      for (let y = 0; y < h; y++) {
        a[y * w + x] = acc / n;
        acc += b[Math.min(y + r + 1, h - 1) * w + x] - b[Math.max(y - r, 0) * w + x];
      }
    }
  }
  return a;
}

// orig, rest: RGBA Uint8ClampedArray of the same w*h crop (rest may be null).
// ox, oy, FW, FH: where the crop sits in the (preview) image, for vignette and radii.
function applyAdjust(orig, rest, w, h, r, ox = 0, oy = 0, FW = w, FH = h) {
  const n = w * h, v = new Float32Array(n * 3), lut = toneLuts(r), g = (k) => adjValue(r, k);
  const st = adjValue(r, 'strength'), longSide = Math.max(FW, FH);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) {
      const a = orig[i * 4 + ch];
      let m = rest ? Math.floor(a + st * (rest[i * 4 + ch] - a) + 0.5) : a;
      m = m < 0 ? 0 : m > 255 ? 255 : m;
      v[i * 3 + ch] = lut[m * 3 + ch];
    }
  }
  const luma = () => { const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = 0.2126 * v[i * 3] + 0.7152 * v[i * 3 + 1] + 0.0722 * v[i * 3 + 2]; return L; };
  const sat = g('saturation'), vib = g('vibrance');
  if (sat || vib) {
    for (let i = 0; i < n * 3; i += 3) {
      const L = 0.2126 * v[i] + 0.7152 * v[i + 1] + 0.0722 * v[i + 2];
      const spread = clamp01(Math.max(v[i], v[i + 1], v[i + 2]) - Math.min(v[i], v[i + 1], v[i + 2]));
      const f = (1 + sat) * (1 + vib * (1 - spread));
      v[i] = L + f * (v[i] - L); v[i + 1] = L + f * (v[i + 1] - L); v[i + 2] = L + f * (v[i + 2] - L);
    }
  }
  const cl = g('clarity');
  if (cl) {
    const L = luma(), base = box3(L, w, h, Math.round(CLARITY_R * longSide));
    for (let i = 0; i < n; i++) {
      const lc = clamp01(L[i]), d = cl * (L[i] - base[i]) * (1 - (2 * lc - 1) ** 2);
      v[i * 3] += d; v[i * 3 + 1] += d; v[i * 3 + 2] += d;
    }
  }
  const dz = g('dehaze');
  if (dz > 0) {
    const dc = new Float32Array(n);
    for (let i = 0; i < n; i++) dc[i] = clamp01(Math.min(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]));
    const haze = box3(dc, w, h, Math.round(DEHAZE_R * longSide));
    for (let i = 0; i < n; i++) {
      const t = Math.max(1 - 0.6 * dz * haze[i], 0.25);
      for (let ch = 0; ch < 3; ch++) v[i * 3 + ch] = 1 - (1 - v[i * 3 + ch]) / t;
    }
  } else if (dz < 0) {
    const k = -0.5 * dz;
    for (let i = 0; i < n * 3; i++) v[i] = v[i] * (1 - k) + 0.85 * k;
  }
  const amt = g('sharpen');
  if (amt) {  // unsharp mask, separable [1 2 1]/4 blur, replicated borders
    const tmp = new Float32Array(n * 3), blur = new Float32Array(n * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const l = y * w + Math.max(x - 1, 0), c0 = y * w + x, rr = y * w + Math.min(x + 1, w - 1);
      for (let ch = 0; ch < 3; ch++) tmp[c0 * 3 + ch] = 0.25 * v[l * 3 + ch] + 0.5 * v[c0 * 3 + ch] + 0.25 * v[rr * 3 + ch];
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const u = Math.max(y - 1, 0) * w + x, c0 = y * w + x, d = Math.min(y + 1, h - 1) * w + x;
      for (let ch = 0; ch < 3; ch++) blur[c0 * 3 + ch] = 0.25 * tmp[u * 3 + ch] + 0.5 * tmp[c0 * 3 + ch] + 0.25 * tmp[d * 3 + ch];
    }
    for (let i = 0; i < n * 3; i++) v[i] += amt * (v[i] - blur[i]);
  }
  const vg = g('vignette');
  if (vg) {
    for (let y = 0; y < h; y++) {
      const yy = (y + oy + 0.5) / FH - 0.5;
      for (let x = 0; x < w; x++) {
        const xx = (x + ox + 0.5) / FW - 0.5, d = Math.sqrt(xx * xx + yy * yy) / Math.SQRT1_2;
        const e = clamp01((d - 0.35) / 0.65), f = 1 - 0.9 * vg * e * e * (3 - 2 * e), i = (y * w + x) * 3;
        v[i] *= f; v[i + 1] *= f; v[i + 2] *= f;
      }
    }
  }
  const out = new ImageData(w, h), o = out.data;
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) o[i * 4 + ch] = Math.floor(clamp01(v[i * 3 + ch]) * 255 + 0.5);
    o[i * 4 + 3] = 255;
  }
  return out;
}

// Auto tone / auto colour: set sliders from the region's current pixels.
function regionPixels(r) {
  const [x0, y0, x1, y1] = r.box, w = x1 - x0, h = y1 - y0;
  const f = Math.min(1, Math.sqrt(4e5 / (w * h))), dw = Math.max(1, Math.round(w * f)), dh = Math.max(1, Math.round(h * f));
  const orig = pixels(photoFor(r), [x0, y0, w, h], dw, dh);
  const rest = r.restorer !== 'none' && r.display ? pixels(r.display, null, dw, dh) : null;
  const st = adjValue(r, 'strength'), px = new Float32Array(dw * dh * 3);
  for (let i = 0; i < dw * dh; i++) for (let ch = 0; ch < 3; ch++) {
    const a = orig[i * 4 + ch];
    px[i * 3 + ch] = (rest ? a + st * (rest[i * 4 + ch] - a) : a) / 255;
  }
  return px;
}

function autoTone(r) {
  const px = regionPixels(r), n = px.length / 3, L = new Float32Array(n);
  for (let i = 0; i < n; i++) L[i] = 0.2126 * px[i * 3] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2];
  const sorted = Float32Array.from(L).sort(), q = (p) => sorted[Math.min(n - 1, Math.floor(p * n))];
  const lo = q(0.005), hi = q(0.995), med = q(0.5);
  const blacks = Math.max(-1, Math.min(1, -lo / 0.1)), whites = Math.max(-1, Math.min(1, (1 - hi) / 0.1));
  const medAfter = clamp01((med - lo) / Math.max(hi - lo, 0.05));
  const exposure = Math.max(-1.5, Math.min(1.5, 0.6 * Math.log2(srgbToLin(0.46) / Math.max(srgbToLin(medAfter), 1e-4))));
  return { blacks: +blacks.toFixed(2), whites: +whites.toFixed(2), exposure: +(Math.round(exposure / 0.05) * 0.05).toFixed(2) };
}

function autoColor(r) {
  // White balance from near-neutral pixels (low saturation, mid brightness), which
  // copes with scenes dominated by one colour; gray world (damped) as a fallback.
  const px = regionPixels(r), n = px.length / 3;
  const acc = (pred) => {
    let R = 0, G = 0, B = 0, c = 0;
    for (let i = 0; i < n; i++) {
      const a = px[i * 3], b = px[i * 3 + 1], d = px[i * 3 + 2], mx = Math.max(a, b, d), mn = Math.min(a, b, d);
      if (mx > 0.97 || mx < 0.08 || !pred(mx, mn)) continue;
      R += srgbToLin(a); G += srgbToLin(b); B += srgbToLin(d); c++;
    }
    return { R, G, B, c };
  };
  let m = acc((mx, mn) => (mx - mn) / mx < 0.2 && mx > 0.15), damp = 0.9;
  if (m.c < 0.02 * n) { m = acc(() => true); damp = 0.5; }
  if (!m.c) return {};
  const t = Math.max(-1, Math.min(1, damp * Math.log2(m.B / m.R) / (2 * WB_STOPS)));
  const rb = (m.R * Math.pow(2, WB_STOPS * t) + m.B * Math.pow(2, -WB_STOPS * t)) / 2;
  const tint = Math.max(-0.6, Math.min(0.6, -damp * Math.log2(rb / m.G) / WB_STOPS));
  return { temperature: +t.toFixed(2), tint: +tint.toFixed(2) };
}

// RGBA pixels of src (or of the window sx,sy,sw,sh of it) resampled to dw x dh.
function pixels(src, win, dw, dh) {
  const c = pixels.c || (pixels.c = document.createElement('canvas'));
  c.width = dw; c.height = dh;
  const t = c.getContext('2d', { willReadFrequently: true });
  if (win) t.drawImage(src, win[0], win[1], win[2], win[3], 0, 0, dw, dh);
  else t.drawImage(src, 0, 0, dw, dh);
  return t.getImageData(0, 0, dw, dh).data;
}

// While a slider is being dragged, big regions are previewed at reduced size.
const DRAFT_PX = 1.2e6;

// What to draw for a region (sized to its box), or null for nothing.
function regionSource(r) {
  const rest = r.restorer !== 'none' ? r.display : null;
  if (r.restorer !== 'none' && !rest) return null;
  if (adjIdentity(r)) return rest;
  const [x0, y0, x1, y1] = r.box, w = x1 - x0, h = y1 - y0;
  const f = S.draft && w * h > DRAFT_PX ? Math.sqrt(DRAFT_PX / (w * h)) : 1;
  const dw = Math.max(1, Math.round(w * f)), dh = Math.max(1, Math.round(h * f));
  const key = JSON.stringify(r.adjust) + '|' + (rest ? rest.src : '') + '|' + r.box.join(',') + '|' + dw;
  if (r.adjKey === key) return r.adjCanvas;
  const origKey = r.box.join(',') + '|' + dw;
  if (r.origKey !== origKey) { r.origPx = pixels(photoFor(r), [x0, y0, w, h], dw, dh); r.origKey = origKey; }
  const out = applyAdjust(r.origPx, rest ? pixels(rest, null, dw, dh) : null, dw, dh, r,
    x0 * f, y0 * f, canvas.width * f, canvas.height * f);
  const c = r.adjCanvas || (r.adjCanvas = document.createElement('canvas'));
  c.width = dw; c.height = dh;
  c.getContext('2d').putImageData(out, 0, 0);
  r.adjKey = key;
  return c;
}

// Removals and replacements (ObjectClear, LaMa, generative fill on a region) change the photo
// itself: the server bakes them into an edited photo that every other layer is processed on.
const isEdit = (r) => r.id !== 'bg' && r.visible && r.restorer !== 'none' && !!(S.restorers[r.restorer] || {}).edits;
const photoFor = (r) => (isEdit(r) ? S.orig : S.base || S.orig);

// The edited photo changed (an edit was added, changed or removed): fetch it and redo the
// layers that were computed on the old one.
async function syncBase() {
  if (syncBase.running) { syncBase.again = true; return; }
  syncBase.running = true;
  try {
    do {
      syncBase.again = false;
      const sid0 = sid();
      const b = await busy('Updating the edited photo…', () => api(`/api/session/${sid0}/base`));
      if (!S.sess || sid() !== sid0) return;
      S.base = b.url ? await loadImg(b.url) : S.orig;
      S.baseLoaded = b.tag;
      for (const r of S.regions) {
        r.origKey = r.adjKey = r.origCropKey = null;
        if (isEdit(r)) r.baked = JSON.stringify(r.values);
      }
      composite();
      for (const r of S.regions) {
        if (isEdit(r) || r.restorer === 'none' || S.baseLoaded !== S.baseTag) continue;
        await sweep(r, r.restorer, r.values, true).catch(() => {});
      }
    } while (syncBase.again || (S.sess && S.baseLoaded !== S.baseTag));
  } catch (_) {
  } finally {
    syncBase.running = false;
    composite();
  }
}

function isTransparent() {
  const bg = region('bg');
  return !!(bg && bg.visible && S.restorers[bg.restorer] && S.restorers[bg.restorer].blend === 'transparent');
}

function origCrop(r) {
  const key = r.box.join(',');
  if (r.origCropKey !== key) {
    const [x0, y0, x1, y1] = r.box, c = r.origCrop || (r.origCrop = document.createElement('canvas'));
    c.width = x1 - x0; c.height = y1 - y0;
    c.getContext('2d').drawImage(photoFor(r), x0, y0, c.width, c.height, 0, 0, c.width, c.height);
    r.origCropKey = key;
  }
  return r.origCrop;
}

// Background effects that depend on the regions above (lens blur) must follow them.
function refreshBackground() {
  const bg = region('bg');
  if (bg && S.restorers[bg.restorer] && S.restorers[bg.restorer].uses_foreground)
    return sweep(bg, bg.restorer, bg.values, true).catch(() => {});
}

async function selectSubject() {
  const info = await busy('Finding the subject…', () => api(`/api/session/${sid()}/subject`, {}));
  const r = await upsertRegion(info);
  renderPanel();
  setActive(r.id, false);
  composite();
  await refreshBackground();
  return r;
}

async function removeBackground() {
  if (!S.regions.some((r) => r.source === 'subject')) await selectSubject();
  await sweep(region('bg'), 'transparent', null);
  toast('Background removed. Export as PNG or WebP to keep the transparency.', true);
}

let compositePending = false;
function scheduleComposite() {
  if (compositePending) return;
  compositePending = true;
  requestAnimationFrame(() => { compositePending = false; composite(); });
}

// ---------------------------------------------------------------- models
const MODEL_STATE = { cuda: 'on GPU', cpu: 'in CPU memory', unloaded: 'not loaded yet', loading: 'loading…', failed: 'failed to load' };
const shortName = (label) => label.split(' · ')[0];

async function refreshModels() {
  let st;
  try { st = await (await fetch('/api/models')).json(); } catch (_) { return false; }
  S.models = Object.fromEntries(st.models.map((m) => [m.key, m]));
  const loading = st.models.find((m) => m.state === 'loading');
  $('#models-chip').classList.toggle('loading', !!loading);
  $('#models-text').textContent = loading ? `Loading ${shortName(loading.label)}…` : `GPU ${st.used_gb.toFixed(1)} / ${st.budget_gb.toFixed(0)} GB`;
  $('#models-mem').textContent = `${st.used_gb.toFixed(1)} of ${st.budget_gb.toFixed(0)} GB used by models`;
  const list = $('#models-list');
  list.innerHTML = '';
  for (const m of st.models) {
    const row = document.createElement('div');
    row.className = 'mp-row';
    row.title = m.error || MODEL_STATE[m.state] || m.state;
    row.innerHTML = `<span class="st ${m.state}"></span><span class="nm">${escapeHtml(m.label)}${m.pinned ? ' <span class="muted">(always on)</span>' : ''}</span>
      <span class="sz">${m.size_gb ? m.size_gb.toFixed(1) + ' GB' : ''}</span><span class="muted">${MODEL_STATE[m.state] || m.state}</span>`;
    list.append(row);
  }
  return !!loading;
}

function pollModels() {
  refreshModels().then((loading) => setTimeout(pollModels, loading ? 1500 : document.hidden ? 20000 : 6000));
}

// A friendly busy label for a restorer that may still have to load.
async function modelLabel(key, fallback) {
  await refreshModels();
  const m = S.models && S.models[key];
  if (!m) return fallback;
  if (m.state === 'unloaded' || m.state === 'loading') return `Loading ${shortName(m.label)} (first use)…`;
  if (m.state === 'cpu') return `Waking ${shortName(m.label)}…`;
  return fallback;
}

// ------------------------------------------------------------ compositing
function composite() {
  if (!S.orig) return;
  const W = canvas.width, H = canvas.height;
  const transparent = isTransparent();
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, W, H);
  if (!transparent || S.compare) ctx.drawImage(S.compare ? S.orig : S.base || S.orig, 0, 0);
  if (!S.compare) {
    for (const r of S.regions) {
      if (!r.visible || !r.box) continue;
      if (transparent && r.id === 'bg') continue;
      // an edit already in the edited photo is not drawn again; while its settings change it is
      if (isEdit(r) && r.baked === JSON.stringify(r.values) && S.base) continue;
      let src = regionSource(r);
      if (!src && transparent) src = origCrop(r);   // untouched regions are still the cut-out
      if (!src) continue;
      const [x0, y0, x1, y1] = r.box;
      if (r.id === 'bg' || (r.blend === 'box' && r.restorer !== 'none')) { ctx.drawImage(src, x0, y0, x1 - x0, y1 - y0); continue; }
      if (!r.maskImg) continue;
      const tmp = composite.tmp || (composite.tmp = document.createElement('canvas'));
      tmp.width = x1 - x0; tmp.height = y1 - y0;
      const t = tmp.getContext('2d');
      t.globalCompositeOperation = 'source-over';
      t.drawImage(src, 0, 0, tmp.width, tmp.height);
      t.globalCompositeOperation = 'destination-in';
      t.drawImage(r.maskImg, 0, 0, tmp.width, tmp.height);
      ctx.drawImage(tmp, x0, y0);
    }
    if (S.split !== null) {
      const sx = Math.round(S.split * W);
      if (sx > 0) ctx.drawImage(S.orig, 0, 0, sx, H, 0, 0, sx, H);
    }
  }
  drawOverlay();
}

// The outline of a region's mask in its colour, about 1.5 screen pixels wide (cached per mask and zoom).
function edgeCanvas(r) {
  const k = S.view.k, key = r.overlayImg.src + '|' + k.toFixed(3) + '|' + colorOf(r);
  if (r._edgeKey === key) return r._edge;
  const w = r.overlayImg.width, h = r.overlayImg.height;
  const d = Math.max(1, 1.5 * (w / canvas.width) / k);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  for (const [dx, dy] of [[d, 0], [-d, 0], [0, d], [0, -d], [d, d], [-d, -d], [d, -d], [-d, d]]) g.drawImage(r.overlayImg, dx, dy);
  g.globalCompositeOperation = 'destination-out';
  g.drawImage(r.overlayImg, 0, 0);
  g.globalCompositeOperation = 'source-in';
  g.fillStyle = colorOf(r);
  g.fillRect(0, 0, w, h);
  r._edge = c; r._edgeKey = key;
  return c;
}

function drawEdge(r, alpha) {
  const { x, y, k } = S.view;
  octx.globalAlpha = alpha;
  octx.imageSmoothingEnabled = true;
  octx.drawImage(edgeCanvas(r), x, y, canvas.width * k, canvas.height * k);
  octx.globalAlpha = 1;
}

function tintRegion(r, alpha) {
  const { x, y, k } = S.view;
  const tint = tintRegion.c || (tintRegion.c = document.createElement('canvas'));
  tint.width = r.overlayImg.width; tint.height = r.overlayImg.height;
  const t = tint.getContext('2d');
  t.globalCompositeOperation = 'source-over';
  t.drawImage(r.overlayImg, 0, 0);
  t.globalCompositeOperation = 'source-in';
  t.fillStyle = colorOf(r);
  t.fillRect(0, 0, tint.width, tint.height);
  octx.globalAlpha = alpha;
  octx.imageSmoothingEnabled = k < 2;
  octx.drawImage(tint, x, y, canvas.width * k, canvas.height * k);
  octx.globalAlpha = 1;
}

// Overlay lives in screen space so markers stay crisp at any zoom.
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (overlay.width !== w * dpr || overlay.height !== h * dpr) { overlay.width = w * dpr; overlay.height = h * dpr; }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, w, h);
  if (!S.orig) return;
  if (S.crop) { drawCrop(w, h); return; }
  const { x, y, k } = S.view;
  // comparing (Space or the split view) shows the photo only, without selection overlays
  const plain = S.compare || S.split !== null;
  const hv = S.hoverRegion && S.hoverRegion !== S.active ? region(S.hoverRegion) : null;
  if (hv && hv.overlayImg && !plain) {
    if (hv.restorer !== 'none') drawEdge(hv, 0.8); else tintRegion(hv, 0.3);
  }
  const r = region(S.active);
  if (r && r.overlayImg && !plain && r.restorer !== 'none') {
    drawEdge(r, 1);   // once a region is processed, a fill would hide the result being tuned
  } else if (r && r.overlayImg && !plain) {
    // Tint the selected mask with its colour.
    const tint = drawOverlay.tint || (drawOverlay.tint = document.createElement('canvas'));
    tint.width = r.overlayImg.width; tint.height = r.overlayImg.height;
    const t = tint.getContext('2d');
    t.globalCompositeOperation = 'source-over';
    t.drawImage(r.overlayImg, 0, 0);
    t.globalCompositeOperation = 'source-in';
    t.fillStyle = colorOf(r);
    t.fillRect(0, 0, tint.width, tint.height);
    octx.globalAlpha = 0.38;
    octx.imageSmoothingEnabled = k < 2;
    octx.drawImage(tint, x, y, canvas.width * k, canvas.height * k);
    octx.globalAlpha = 1;
    for (const [px, py, l] of r.points || []) {
      const sx = x + px * k, sy = y + py * k;
      octx.beginPath(); octx.arc(sx, sy, 6, 0, Math.PI * 2);
      octx.fillStyle = l ? '#2ecc71' : '#ff5a5a'; octx.fill();
      octx.lineWidth = 2; octx.strokeStyle = '#fff'; octx.stroke();
      octx.beginPath();
      octx.moveTo(sx - 3, sy); octx.lineTo(sx + 3, sy);
      if (l) { octx.moveTo(sx, sy - 3); octx.lineTo(sx, sy + 3); }
      octx.lineWidth = 1.6; octx.stroke();
    }
    if (r.boxPrompt) {
      const [bx0, by0, bx1, by1] = r.boxPrompt;
      octx.setLineDash([5, 4]); octx.strokeStyle = colorOf(r); octx.lineWidth = 1.5;
      octx.strokeRect(x + bx0 * k, y + by0 * k, (bx1 - bx0) * k, (by1 - by0) * k);
      octx.setLineDash([]);
    }
  }
  if (S.drag && S.drag.type === 'brush') {
    const d = S.drag, rr = region(S.editing);
    octx.strokeStyle = d.add ? (rr ? colorOf(rr) : '#5b9dff') : '#ff5a5a';
    octx.globalAlpha = 0.45;
    octx.lineWidth = 2 * d.radius * k;
    octx.lineCap = octx.lineJoin = 'round';
    octx.beginPath();
    d.pts.forEach(([px, py], i) => (i ? octx.lineTo(x + px * k, y + py * k) : octx.moveTo(x + px * k, y + py * k)));
    if (d.pts.length === 1) octx.lineTo(x + d.pts[0][0] * k + 0.1, y + d.pts[0][1] * k);
    octx.stroke();
    octx.globalAlpha = 1;
  }
  if (S.tool === 'brush' && S.hover) {
    octx.beginPath(); octx.arc(S.hover.sx, S.hover.sy, S.brush, 0, Math.PI * 2);
    octx.lineWidth = 1.5; octx.strokeStyle = '#fff'; octx.stroke();
    octx.beginPath(); octx.arc(S.hover.sx, S.hover.sy, S.brush + 1.5, 0, Math.PI * 2);
    octx.lineWidth = 1; octx.strokeStyle = 'rgba(0,0,0,.6)'; octx.stroke();
  }
  if (S.drag && S.drag.type === 'box') {
    const { x0, y0, x1, y1 } = S.drag;
    octx.setLineDash([5, 4]); octx.strokeStyle = '#fff'; octx.lineWidth = 1.5;
    octx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    octx.setLineDash([]);
  }
  if (S.split !== null && !S.compare) {
    const sx = x + S.split * canvas.width * k;
    const top = Math.max(0, y), bot = Math.min(h, y + canvas.height * k);
    octx.strokeStyle = '#fff'; octx.lineWidth = 2;
    octx.beginPath(); octx.moveTo(sx, top); octx.lineTo(sx, bot); octx.stroke();
    const my = (top + bot) / 2;
    octx.fillStyle = '#fff';
    octx.beginPath(); octx.arc(sx, my, 13, 0, Math.PI * 2); octx.fill();
    octx.fillStyle = '#222';
    octx.beginPath(); octx.moveTo(sx - 3, my - 5); octx.lineTo(sx - 8, my); octx.lineTo(sx - 3, my + 5); octx.fill();
    octx.beginPath(); octx.moveTo(sx + 3, my - 5); octx.lineTo(sx + 8, my); octx.lineTo(sx + 3, my + 5); octx.fill();
    octx.font = '600 11px system-ui'; octx.fillStyle = 'rgba(0,0,0,.55)';
    const lab = (t, tx) => { const w = octx.measureText(t).width + 10; octx.fillStyle = 'rgba(0,0,0,.6)'; octx.fillRect(tx, top + 8, w, 18); octx.fillStyle = '#fff'; octx.fillText(t, tx + 5, top + 21); };
    lab('Original', sx - octx.measureText('Original').width - 22);
    lab('Restored', sx + 12);
  }
}

// ------------------------------------------------------------------ crop
// Mirrors rap/geometry.py: flip, clockwise quarter turns, straighten (clockwise,
// canvas grows to the rotated bounding box), crop in normalised rotated-frame coords.
function cropFrame() {
  const c = S.crop, odd = c.rot % 2 !== 0;
  const w = odd ? canvas.height : canvas.width, h = odd ? canvas.width : canvas.height;   // after quarter turns
  const a = (c.angle * Math.PI) / 180, co = Math.abs(Math.cos(a)), si = Math.abs(Math.sin(a));
  return { w, h, W: w * co + h * si, H: w * si + h * co, co, si };
}

function cropRatio() {
  const v = $('#crop-ratio').value, f = cropFrame();
  if (v === 'free') return null;
  if (v === 'original') return f.w / f.h;
  return parseFloat(v);
}

// Largest centred rectangle of the given aspect (or the frame's) inside the rotated photo.
function defaultCropRect() {
  const f = cropFrame(), r = cropRatio() || f.w / f.h;
  // (a hair smaller when rotated, so edge pixels blended with the empty corners stay out)
  let w = Math.min(f.w / (f.co + f.si / r), f.h / (f.si + f.co / r)) * (S.crop.angle ? 0.99 : 1);
  let h = w / r;
  return [(f.W - w) / 2 / f.W, (f.H - h) / 2 / f.H, (f.W + w) / 2 / f.W, (f.H + h) / 2 / f.H];
}

function cropLayout() {
  const f = cropFrame(), vw = viewport.clientWidth, vh = viewport.clientHeight;
  const k = Math.min((vw - 80) / f.W, (vh - 150) / f.H);
  return { f, k, cx: vw / 2, cy: (vh - 70) / 2 + 12 };
}

function cropRectScreen() {
  const { f, k, cx, cy } = cropLayout(), r = S.crop.rect;
  const X = (u) => cx + (u - 0.5) * f.W * k, Y = (v) => cy + (v - 0.5) * f.H * k;
  return [X(r[0]), Y(r[1]), X(r[2]), Y(r[3])];
}

function enterCrop() {
  if (!S.orig) return;
  doneEditing();
  S.crop = { rot: 0, flipH: false, flipV: false, angle: 0, rect: [0, 0, 1, 1], drag: null };
  $('#crop-ratio').value = 'free';
  $('#crop-angle').value = 0;
  $('#cropbar').hidden = false;
  stage.style.visibility = 'hidden';
  cropChanged();
}

function exitCrop() {
  S.crop = null;
  $('#cropbar').hidden = true;
  stage.style.visibility = '';
  drawOverlay();
}

function cropChanged() {
  S.crop.rect = defaultCropRect();
  $('#crop-angle-val').textContent = S.crop.angle.toFixed(1) + '°';
  drawOverlay();
}

function cropOutsidePhoto() {
  // true if the rectangle includes corners that were never part of the photo
  const c = S.crop;
  if (Math.abs(c.angle) < 1e-3) return false;
  const f = cropFrame(), r = c.rect, a = (c.angle * Math.PI) / 180;
  const pts = [[r[0], r[1]], [r[2], r[1]], [r[0], r[3]], [r[2], r[3]]];
  return pts.some(([u, v]) => {
    const x = (u - 0.5) * f.W, y = (v - 0.5) * f.H;   // rotate back into the photo frame
    const px = x * Math.cos(-a) - y * Math.sin(-a), py = x * Math.sin(-a) + y * Math.cos(-a);
    return Math.abs(px) > f.w / 2 + 0.5 || Math.abs(py) > f.h / 2 + 0.5;
  });
}

function drawCrop(w, h) {
  const c = S.crop, { f, k, cx, cy } = cropLayout();
  octx.save();
  octx.translate(cx, cy);
  octx.rotate(((c.rot * 90 + c.angle) * Math.PI) / 180);
  octx.scale(c.flipH ? -1 : 1, c.flipV ? -1 : 1);
  octx.imageSmoothingQuality = 'high';
  octx.drawImage(canvas, (-canvas.width / 2) * k, (-canvas.height / 2) * k, canvas.width * k, canvas.height * k);
  octx.restore();
  const [x0, y0, x1, y1] = cropRectScreen();
  octx.fillStyle = 'rgba(12,13,15,.62)';
  octx.beginPath(); octx.rect(0, 0, w, h); octx.rect(x0, y0, x1 - x0, y1 - y0); octx.fill('evenodd');
  octx.strokeStyle = 'rgba(255,255,255,.35)'; octx.lineWidth = 1;
  for (const t of [1 / 3, 2 / 3]) {
    octx.beginPath(); octx.moveTo(x0 + (x1 - x0) * t, y0); octx.lineTo(x0 + (x1 - x0) * t, y1);
    octx.moveTo(x0, y0 + (y1 - y0) * t); octx.lineTo(x1, y0 + (y1 - y0) * t); octx.stroke();
  }
  octx.strokeStyle = '#fff'; octx.lineWidth = 1.5; octx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  octx.fillStyle = '#fff';
  for (const [hx, hy] of cropHandles()) octx.fillRect(hx[1] - 5, hy[1] - 5, 10, 10);
  const note = cropOutsidePhoto();
  $('#crop-note').hidden = !note;
  const pw = Math.round((c.rect[2] - c.rect[0]) * f.W / S.sess.scale), ph = Math.round((c.rect[3] - c.rect[1]) * f.H / S.sess.scale);
  octx.font = '600 11px system-ui'; const label = `${pw} × ${ph}`, lw = octx.measureText(label).width + 12;
  octx.fillStyle = 'rgba(0,0,0,.65)'; octx.fillRect(x0, y0 - 24, lw, 20); octx.fillStyle = '#fff'; octx.fillText(label, x0 + 6, y0 - 10);
}

function cropHandles() {
  const [x0, y0, x1, y1] = cropRectScreen(), mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return [[['nw', x0], ['nw', y0]], [['ne', x1], ['ne', y0]], [['sw', x0], ['sw', y1]], [['se', x1], ['se', y1]],
          [['n', mx], ['n', y0]], [['s', mx], ['s', y1]], [['w', x0], ['w', my]], [['e', x1], ['e', my]]];
}

function cropHit(p) {
  for (const [hx, hy] of cropHandles()) if (Math.abs(p.sx - hx[1]) < 10 && Math.abs(p.sy - hy[1]) < 10) return hx[0];
  const [x0, y0, x1, y1] = cropRectScreen();
  return p.sx > x0 && p.sx < x1 && p.sy > y0 && p.sy < y1 ? 'move' : null;
}

function cropDrag(d, p) {
  const { f, k } = cropLayout(), r = [...d.rect0], ratio = cropRatio();
  const du = (p.sx - d.sx) / (f.W * k), dv = (p.sy - d.sy) / (f.H * k), MIN = 0.03;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  if (d.mode === 'move') {
    const w = r[2] - r[0], h = r[3] - r[1];
    const x0 = Math.max(0, Math.min(1 - w, r[0] + du)), y0 = Math.max(0, Math.min(1 - h, r[1] + dv));
    S.crop.rect = [x0, y0, x0 + w, y0 + h];
    return;
  }
  if (d.mode.includes('w')) r[0] = clamp(Math.min(r[0] + du, r[2] - MIN));
  if (d.mode.includes('e')) r[2] = clamp(Math.max(r[2] + du, r[0] + MIN));
  if (d.mode.includes('n')) r[1] = clamp(Math.min(r[1] + dv, r[3] - MIN));
  if (d.mode.includes('s')) r[3] = clamp(Math.max(r[3] + dv, r[1] + MIN));
  if (ratio) {  // keep the aspect: derive the height from the width (pixel units)
    const wpx = (r[2] - r[0]) * f.W;
    let hn = wpx / ratio / f.H;
    if (d.mode === 'n' || d.mode === 's') {  // edge drag: width follows height
      const wn = ((r[3] - r[1]) * f.H * ratio) / f.W, mx = (r[0] + r[2]) / 2;
      r[0] = mx - wn / 2; r[2] = mx + wn / 2;
    } else if (d.mode.includes('n')) r[1] = r[3] - hn;
    else if (d.mode.includes('s') || d.mode === 'e' || d.mode === 'w') {
      if (d.mode === 'e' || d.mode === 'w') { const my = (r[1] + r[3]) / 2; r[1] = my - hn / 2; r[3] = my + hn / 2; }
      else r[3] = r[1] + hn;
    }
    if (r[0] < 0 || r[1] < 0 || r[2] > 1 || r[3] > 1) return;   // would leave the frame: ignore this step
  }
  S.crop.rect = r;
}

async function applyCrop() {
  const c = S.crop;
  const body = { rot: ((c.rot % 4) + 4) % 4, flip_h: c.flipH, flip_v: c.flipV, angle: c.angle, crop: c.rect, fill: true };
  const info = await busy(cropOutsidePhoto() ? 'Cropping and filling corners…' : 'Cropping…', () =>
    api(`/api/session/${sid()}/transform`, body));
  exitCrop();
  setTool('point');
  await openSession(info);
  toast('Cropped (Ctrl+Z to undo).', true);
}

// ------------------------------------------------------------------ view
function applyView() {
  const { x, y, k } = S.view;
  stage.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
  stage.classList.toggle('pixelated', k >= 2);
  $('#zoom-label').textContent = Math.round(k * 100) + '%';
  drawOverlay();
}

function fit() {
  if (!S.orig) return;
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  const k = Math.min((vw - 40) / canvas.width, (vh - 40) / canvas.height, 4);
  S.view = { k, x: (vw - canvas.width * k) / 2, y: (vh - canvas.height * k) / 2 };
  applyView();
}

function zoomAt(sx, sy, k) {
  k = Math.max(0.05, Math.min(32, k));
  const v = S.view;
  v.x = sx - (sx - v.x) * (k / v.k);
  v.y = sy - (sy - v.y) * (k / v.k);
  v.k = k;
  applyView();
}

function toImage(e) {
  const rc = viewport.getBoundingClientRect();
  const sx = e.clientX - rc.left, sy = e.clientY - rc.top;
  return { sx, sy, ix: (sx - S.view.x) / S.view.k, iy: (sy - S.view.y) / S.view.k };
}

const inImage = (p) => p.ix >= 0 && p.iy >= 0 && p.ix < canvas.width && p.iy < canvas.height;

function nearSplit(p) {
  if (S.split === null) return false;
  return Math.abs(p.sx - (S.view.x + S.split * canvas.width * S.view.k)) < 14;
}

// --------------------------------------------------------------- session
async function openSession(info, keepView = false) {
  S.sess = info;
  try { history.replaceState(null, '', '#s=' + info.id); } catch (_) {}
  $('#btn-undo-all').disabled = !info.can_undo;
  $('#btn-redo-all').disabled = !info.can_redo;
  S.orig = await loadImg(info.image_url);
  S.base = info.base_url ? await loadImg(info.base_url) : S.orig;   // the photo with removals/replacements applied
  S.baseTag = S.baseLoaded = info.base_tag;
  canvas.width = info.preview[0];
  canvas.height = info.preview[1];
  S.regions = [];
  S.active = null;
  S.editing = null;
  S.split = null;
  for (const r of info.regions) await upsertRegion(r);
  for (const r of S.regions) if (isEdit(r)) r.baked = JSON.stringify(r.values);
  $('#empty').hidden = true;
  $('#toolbar').hidden = false;
  for (const el of ['#btn-auto', '#btn-apply', '#btn-export', '#btn-compare', '#btn-split', '#find-input', '#find button', '#btn-subject', '#btn-removebg']) $(el).disabled = false;
  exportUpdate();
  $('#btn-split').classList.remove('active');
  $('#hint').innerHTML = 'Click an object (or paint, or find it by name), then pick a model for it. ' +
    'Everything else is the <b>Background</b> layer. <b>Auto enhance</b> does faces + background in one click.' +
    '<br>Hold <kbd>Space</kbd> to compare · <kbd>?</kbd> for all shortcuts.';
  if (info.downscaled_from && !keepView) {
    const [ow, oh] = info.downscaled_from;
    toast(`Large photo (${ow} × ${oh}): working at ${info.width} × ${info.height}, the most this tool handles. Export uses this size (or upscale it).`, true, 8000);
  }
  setTool(S.tool === 'crop' ? 'point' : S.tool);   // refreshes the mode hint
  if (keepView) applyView(); else fit();
  renderPanel();
  composite();
}

async function upload(file) {
  if (!file) return;
  if (S.dirty && !confirm('Open a new image? Unsaved changes to the current one will be lost.')) return;
  const fd = new FormData();
  fd.append('file', file);
  const info = await busy('Uploading…', () => api('/api/session', fd));
  S.dirty = false; S.tuned = false; S.exported = false;
  await openSession(info);
  analyze();
}

async function openExample(name) {
  const info = await busy('Opening…', () => api('/api/session/example/' + encodeURIComponent(name), {}));
  S.dirty = false; S.tuned = false; S.exported = false;
  await openSession(info);
  analyze();
}

// Back to the start page. The session stays on the server (reload the old URL to resume).
function goHome() {
  if (!S.sess) return;
  if (S.dirty && !confirm('Go back to the start page? Your edits are kept on the server until you open another image, but unsaved results are easy to lose.')) return;
  if (S.crop) exitCrop();
  doneEditing();
  S.sess = null; S.orig = null; S.base = null; S.baseTag = S.baseLoaded = null; S.regions = []; S.active = S.editing = null; S.dirty = false; S.split = null;
  try { history.replaceState(null, '', location.pathname); } catch (_) {}
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawOverlay();
  $('#empty').hidden = false;
  $('#empty').scrollTop = 0;
  $('#toolbar').hidden = true;
  $('#modehint').hidden = true;
  $('#steps').hidden = true;
  $('#suggest').hidden = true;
  $('#regions').innerHTML = '';
  $('#hint').hidden = false;
  $('#hint').textContent = 'Open an image to start.';
  for (const el of ['#btn-auto', '#btn-apply', '#btn-export', '#btn-compare', '#btn-split', '#find-input', '#find button',
                    '#btn-subject', '#btn-removebg', '#btn-undo-all', '#btn-redo-all']) $(el).disabled = true;
}

// After a reload: pick the session up again from the URL, results included.
async function reattach() {
  const m = location.hash.match(/s=([0-9a-f]+)/);
  if (!m) return;
  let info;
  try { info = await api('/api/session/' + m[1]); } catch (_) { history.replaceState(null, '', location.pathname); return; }
  await openSession(info);
  for (const r of S.regions) if (r.restorer !== 'none') await sweep(r, r.restorer, r.values, true).catch(() => {});
  composite();
  toast('Welcome back: your edits were restored.', true);
}

async function upsertRegion(info) {
  let r = region(info.id);
  if (!r) {
    r = { restorer: 'none', values: {}, visible: true, cands: [], display: null };
    S.regions.push(r);
  }
  const keepBox = r.boxPrompt;
  Object.assign(r, info);
  r.boxPrompt = keepBox;
  if (info.mask_url) [r.maskImg, r.overlayImg] = await Promise.all([loadImg(info.mask_url), loadImg(info.overlay_url)]);
  return r;
}

// ------------------------------------------------------------ selection
async function clickPrompt(p, negative) {
  let r = region(S.editing);
  if (r && r.id === 'bg') r = null;
  if (!r && negative) { toast('Select something first; right-click removes parts of the current selection.'); return; }
  const points = r ? [...(r.points || []), [p.ix, p.iy, negative ? 0 : 1]] : [[p.ix, p.iy, 1]];
  const body = { region_id: r ? r.id : null, points, box: r ? r.boxPrompt || null : null };
  const info = await busy('Segmenting…', () => api(`/api/session/${sid()}/click`, body));
  const nr = await upsertRegion(info);
  setActive(nr.id);
  await afterMaskChange(nr);
}

async function boxPrompt(b) {
  let r = region(S.editing);
  if (r && r.id === 'bg') r = null;
  const body = { region_id: r ? r.id : null, points: r ? r.points || [] : [], box: b };
  const info = await busy('Segmenting…', () => api(`/api/session/${sid()}/click`, body));
  const nr = await upsertRegion(info);
  nr.boxPrompt = b;
  setActive(nr.id);
  await afterMaskChange(nr);
}

async function brushStroke(d) {
  let r = region(S.editing) || region(S.active);
  if (r && r.id === 'bg') r = null;
  if (!r && !d.add) { toast('Select a region first; right-drag erases from the current selection.'); return; }
  const info = await busy(d.add ? 'Painting…' : 'Erasing mask…', () =>
    api(`/api/session/${sid()}/paint`, { region_id: r ? r.id : null, points: d.pts, radius: d.radius, add: d.add }));
  const nr = await upsertRegion(info);
  setActive(nr.id);
  await afterMaskChange(nr);
}

async function historyStep(dir) {
  if (!S.sess) return;
  const info = await busy(dir === 'undo' ? 'Undoing…' : 'Redoing…', () => api(`/api/session/${sid()}/${dir}`, {}));
  const keep = S.view, active = S.active, editing = S.editing;
  await openSession(info, true);
  S.view = keep; applyView();
  // stay on the same region if it still exists
  if (active && region(active)) setActive(active, editing === active);
  // Restored regions still need their results in the browser (cached server-side).
  for (const r of S.regions) if (r.restorer !== 'none') await sweep(r, r.restorer, r.values, true);
  composite();
}

async function findText(prompt, separate) {
  const out = await busy(`Finding “${prompt}”…`, () => api(`/api/session/${sid()}/text`, { prompt, separate }));
  let last = null;
  for (const info of out.regions) last = await upsertRegion(info);
  toast(`Found ${out.count} × “${prompt}”`, true);
  doneEditing();
  renderPanel();
  refreshBackground();
  if (last) setActive(last.id, false);
  composite();
}

async function afterMaskChange(r) {
  renderPanel();
  updateSteps();
  if (r.restorer !== 'none') await sweep(r, r.restorer, r.values);
  composite();
  await refreshBackground();
}

function setActive(id, editing = true) {
  S.active = id;
  const r = region(id);
  const show = editing && r && r.id !== 'bg';
  S.editing = show ? id : null;
  $('#editing').hidden = !show;
  if (show) {
    $('#editing-name').textContent = r.name;
    $('#editing .dot').style.background = colorOf(r);
  }
  document.querySelectorAll('.card').forEach((c) => c.classList.toggle('active', c.dataset.id === id));
  drawOverlay();
}

function doneEditing() {
  S.active = null;
  S.editing = null;
  $('#editing').hidden = true;
  document.querySelectorAll('.card').forEach((c) => c.classList.remove('active'));
  drawOverlay();
}

// ------------------------------------------------------------ restoration
function autoEnhance() {
  // One outer busy state so the indicator stays up between the steps.
  return busy('Auto enhance…', autoEnhanceSteps);
}

async function autoEnhanceSteps() {
  const bg = region('bg');
  if (S.restorers.pisasr) await sweep(bg, 'pisasr', null);
  if (!S.restorers.codeformer) return;
  let found = null;
  try {
    found = await busy('Looking for faces…', () => api(`/api/session/${sid()}/text`, { prompt: 'face', separate: false }));
  } catch (_) { toast('Background restored; no faces found.', true); return; }
  const r = await upsertRegion(found.regions[0]);
  r.name = found.count > 1 ? `${found.count} faces` : 'Face';
  renderPanel();
  await sweep(r, 'codeformer', { w: 0.7 }, true);
  api(`/api/session/${sid()}/region/${r.id}`, { name: r.name }, 'PATCH').catch(() => {});
  toast(`Restored the background and ${found.count} face${found.count > 1 ? 's' : ''}. Tweak any slider to taste.`, true);
}

async function sweep(r, key, values, forceValues = false) {
  r.busy = true;
  updateCardStatus(r);
  try {
    const spec = S.restorers[key];
    const verb = key === 'objectclear' ? 'Removing…' : key === 'genfill' ? 'Generating…'
      : spec && spec.group === 'remove' ? 'Erasing…' : spec && spec.group === 'effect' ? 'Applying…' : 'Restoring…';
    const label = await modelLabel(key, verb);
    const out = await busy(label, () =>
      api(`/api/session/${sid()}/region/${r.id}/sweep`, { restorer: key, values: forceValues || key === r.restorer ? values : null }));
    r.restorer = out.restorer;
    r.blend = out.blend || 'mask';
    r.values = out.values || {};
    r.estimate = out.estimate || {};
    r.info = out.info || '';
    r.box = out.box;
    r.cands = await Promise.all((out.candidates || []).map(async (c) => ({ value: c.value, img: await loadImg(c.url) })));
    r.display = out.current ? await loadImg(out.current) : null;
    r.displayKey = JSON.stringify(r.values);
  } finally {
    r.busy = false;
    renderCard(r);
    composite();
  }
}

// Called while dragging the primary slider: show the closest precomputed result.
function previewPrimary(r, v) {
  if (!r.cands.length) return;
  let best = r.cands[0];
  for (const c of r.cands) if (Math.abs(c.value - v) < Math.abs(best.value - v)) best = c;
  r.display = best.img;
  composite();
}

async function renderExact(r) {
  const key = JSON.stringify(r.values);
  const token = (r.renderToken = (r.renderToken || 0) + 1);
  r.busy = true;
  updateCardStatus(r);
  try {
    const out = await api(`/api/session/${sid()}/region/${r.id}/render`, { values: r.values });
    const img = await loadImg(out.url);
    if (token === r.renderToken) { r.display = img; r.displayKey = key; composite(); }
  } catch (e) { toast(e.message); }
  finally { if (token === r.renderToken) { r.busy = false; updateCardStatus(r); } }
}

// ------------------------------------------------------------ guidance
const STEPS_KEY = 'rap.steps.hidden';
function updateSteps() {
  let hidden = false;
  try { hidden = localStorage.getItem(STEPS_KEY) === '1'; } catch (_) {}
  $('#steps').hidden = hidden || !S.sess;
  $('#hint').hidden = !!S.sess && !$('#steps').hidden;
  if ($('#steps').hidden) return;
  const regs = S.regions.filter((r) => r.id !== 'bg');
  const done = {
    select: regs.length > 0 || region('bg').restorer !== 'none',
    choose: S.regions.some((r) => r.restorer !== 'none' || !adjIdentity({ ...r, restorer: 'none' })),
    tune: !!S.tuned,
    export: !!S.exported,
  };
  document.querySelectorAll('#steps li').forEach((li) => li.classList.toggle('done', !!done[li.dataset.step]));
}

async function analyze() {
  const sid0 = sid();
  $('#suggest').hidden = false;
  $('#suggest-list').innerHTML = '';
  $('#suggest-status').textContent = 'looking at the photo…';
  let out;
  try { out = await api(`/api/session/${sid0}/analyze`, {}); } catch (_) { $('#suggest').hidden = true; return; }
  if (!S.sess || sid() !== sid0) return;
  $('#suggest-status').textContent = '';
  if (!out.suggestions.length) { $('#suggest').hidden = true; return; }
  for (const sg of out.suggestions) {
    const card = document.createElement('div');
    card.className = 'sg-card';
    card.innerHTML = `<div class="sg-text"><div class="sg-title">${escapeHtml(sg.title)}</div><div class="sg-detail">${escapeHtml(sg.detail)}</div></div>
      <button class="btn small primary">${sg.action.type === 'export' ? 'Set' : 'Do it'}</button><button class="sg-x" title="Dismiss">×</button>`;
    card.querySelector('.sg-x').onclick = () => { card.remove(); if (!$('#suggest-list').children.length) $('#suggest').hidden = true; };
    card.querySelector('.btn').onclick = async () => {
      await runSuggestion(sg.action).catch(() => {});
      card.classList.add('done');
      card.querySelector('.btn').textContent = '✓';
      card.querySelector('.btn').disabled = true;
    };
    $('#suggest-list').append(card);
  }
}

async function runSuggestion(a) {
  if (a.type === 'sweep') await sweep(region(a.region), a.restorer, null);
  else if (a.type === 'auto') await autoEnhance();
  else if (a.type === 'portrait') {
    if (!S.regions.some((r) => r.source === 'subject')) await selectSubject();
    await sweep(region('bg'), 'lens_blur', null);
  } else if (a.type === 'export') {
    $('#export-size').value = a.size;
    exportUpdate();
    $('#export-panel').hidden = false;
    toast(`Export size set to ${a.size}×. Press Download when you are ready.`, true);
  }
}

// ----------------------------------------------------------------- panel
// Task-first names for the picker; the model name is shown in small print.
const TASKS = {
  none: ['No change', 'Leave this region as it is'],
  pisasr: ['Enhance photo', 'Sharpen, denoise and clean up real photos'],
  nafnet_deblur: ['Fix motion blur', 'Camera shake or moving subjects'],
  fbcnn_deblock: ['Fix JPEG artifacts', 'Blocky, ringing compression'],
  scunet: ['Remove noise', 'Real grain: phones, high ISO, night shots'],
  drunet: ['Remove noise · set strength', 'Starts at an estimate for this region'],
  codeformer: ['Restore faces', 'Blurry, damaged or tiny faces'],
  lama: ['Erase', 'Scratches, text, wires, small objects'],
  objectclear: ['Remove object + shadow', 'Takes the shadow and reflection with it'],
  genfill: ['Replace with…', 'Generates what you describe'],
  ddcolor: ['Colorize', 'Black-and-white to colour'],
  cidnet: ['Brighten dark photo', 'Under-exposed or night shots'],
  lens_blur: ['Blur background', 'Portrait look, based on depth'],
  blur: ['Blur (privacy)', 'Hide faces, plates, screens'],
  pixelate: ['Pixelate (privacy)', 'Mosaic over faces, plates, screens'],
  color_fill: ['Solid color', 'A flat colour instead of the background'],
  transparent: ['Transparent', 'Cut out, transparent background'],
};
const SECTIONS = [
  ['Fix', ['pisasr', 'scunet', 'drunet', 'nafnet_deblur', 'fbcnn_deblock']],
  ['Faces', ['codeformer']],
  ['Remove & replace', ['lama', 'objectclear', 'genfill']],
  ['Color & light', ['ddcolor', 'cidnet']],
  ['Effects', ['lens_blur', 'blur', 'pixelate', 'color_fill', 'transparent']],
];
const taskName = (key) => (TASKS[key] || [S.restorers[key] ? S.restorers[key].label : key])[0];

function offeredFor(r, key) {
  const spec = S.restorers[key];
  if (!spec) return false;
  if (spec.needs_mask && r.id === 'bg' && !spec.allow_bg) return false;   // erasers need a selected region
  if (spec.bg_only && r.id !== 'bg') return false;                       // background-only effects
  return true;
}

function suggestedFor(r) {
  if (r.id === 'bg') return 'pisasr';
  if (/face/i.test(r.name)) return 'codeformer';
  return 'pisasr';
}

function buildPickButton(r) {
  const b = document.createElement('button');
  b.className = 'pick-btn';
  const spec = S.restorers[r.restorer];
  const model = r.restorer !== 'none' && spec && spec.group !== 'effect' ? shortName(spec.label) : '';
  b.innerHTML = `<span class="pk-name">${escapeHtml(taskName(r.restorer))}</span>
    <span class="pk-model">${escapeHtml(model)}</span><span class="pk-caret">▾</span>`;
  b.title = 'Choose what to do with this region';
  b.onclick = (e) => { e.stopPropagation(); openPicker(r, b); };
  return b;
}

function openPicker(r, anchor) {
  const pk = $('#picker');
  pk.innerHTML = '';
  const rec = suggestedFor(r);
  const add = (key) => {
    const [name, sub] = TASKS[key] || [S.restorers[key].label, ''];
    const it = document.createElement('button');
    it.className = 'pk-item' + (key === r.restorer ? ' on' : '');
    const model = key !== 'none' && S.restorers[key].group !== 'effect' ? shortName(S.restorers[key].label) : '';
    it.innerHTML = `<span class="pk-top"><span class="pk-name">${escapeHtml(name)}</span>
      ${key === rec && key !== r.restorer ? '<span class="pk-rec">Recommended</span>' : ''}
      <span class="pk-model">${escapeHtml(model)}</span></span><span class="pk-sub">${escapeHtml(sub)}</span>`;
    it.onclick = () => { closePicker(); chooseRestorer(r, key); };
    return it;
  };
  pk.append(add('none'));
  const known = new Set(['none'].concat(...SECTIONS.map(([, ks]) => ks)));
  const extra = S.order.filter((k) => !known.has(k));
  for (const [title, keys] of SECTIONS.concat(extra.length ? [['Other', extra]] : [])) {
    const ks = keys.filter((k) => offeredFor(r, k));
    if (!ks.length) continue;
    const h = document.createElement('div');
    h.className = 'pk-sec';
    h.textContent = title;
    pk.append(h, ...ks.map(add));
  }
  const rc = anchor.getBoundingClientRect();
  pk.hidden = false;
  const top = Math.min(rc.bottom + 4, window.innerHeight - pk.offsetHeight - 8);
  pk.style.top = Math.max(8, top) + 'px';
  pk.style.left = Math.max(8, Math.min(rc.left, window.innerWidth - pk.offsetWidth - 8)) + 'px';
}

function closePicker() { $('#picker').hidden = true; }

function chooseRestorer(r, key) {
  const sp = S.restorers[key];
  if (sp && sp.prompt && !(r.restorer === key && r.values.prompt)) {
    // wait for a prompt before generating anything
    r.restorer = key; r.values = {}; r.cands = []; r.display = null; r.blend = sp.blend;
    renderCard(r); composite();
    const inp = document.querySelector(`.card[data-id="${r.id}"] input.prompt`);
    if (inp) inp.focus();
    return;
  }
  sweep(r, key, null);
}

const GROUP_LABELS = { restore: 'Restore', face: 'Faces', remove: 'Remove', generate: 'Generate', color: 'Colorize', light: 'Light', effect: 'Effects' };

const ICON = {
  eye: '<svg viewBox="0 0 24 24"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  edit: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16v4z"/></svg>',
  up: '<svg viewBox="0 0 24 24"><path d="M6 15l6-6 6 6"/></svg>',
};

function renderPanel() {
  updateSteps();
  const box = $('#regions');
  box.innerHTML = '';
  // Top-most region first, background last (like a layers panel).
  for (const r of [...S.regions].reverse()) box.appendChild(buildCard(r));
}

function renderCard(r) {
  const old = document.querySelector(`.card[data-id="${r.id}"]`);
  if (old) old.replaceWith(buildCard(r));
}

function updateCardStatus(r) {
  const el = document.querySelector(`.card[data-id="${r.id}"] .status`);
  if (el) el.hidden = !r.busy;
}

function buildCard(r) {
  const el = document.createElement('div');
  el.className = 'card' + (r.id === 'bg' ? ' bg' : '') + (S.active === r.id ? ' active' : '');
  el.dataset.id = r.id;
  const spec = S.restorers[r.restorer];

  const head = document.createElement('div');
  head.className = 'card-head';
  head.innerHTML = `<span class="swatch" style="background:${colorOf(r)}"></span>
    <input class="name" type="text" value="${escapeHtml(r.name)}" ${r.id === 'bg' ? 'readonly' : ''}>
    <span class="spinner status" hidden></span>`;
  if (r.id !== 'bg') {
    head.append(iconBtn(ICON.up, 'Move up (draw on top)', () => moveRegion(r, +1)));
    head.append(iconBtn(ICON.edit, 'Edit selection', () => setActive(r.id)));
  }
  head.append(iconBtn(ICON.eye, r.visible ? 'Hide restoration' : 'Show restoration', () => toggleVisible(r), !r.visible));
  if (r.id !== 'bg') head.append(iconBtn(ICON.trash, 'Delete region', () => deleteRegion(r)));
  head.querySelector('.swatch').onclick = () => setActive(r.id, r.id !== 'bg');
  const name = head.querySelector('.name');
  name.onchange = () => patchRegion(r, { name: name.value });
  el.append(head);

  const body = document.createElement('div');
  body.className = 'card-body';
  if (r.id === 'bg') {
    const d = document.createElement('div');
    d.className = 'desc';
    d.textContent = 'Everything not covered by a region above.';
    body.append(d);
  }
  body.append(buildPickButton(r));
  if (spec && spec.key !== 'none') {
    const d = document.createElement('div');
    d.className = 'desc';
    d.textContent = spec.description;
    body.append(d);
    if (spec.prompt) {
      const row = document.createElement('div');
      row.className = 'prompt-row';
      row.innerHTML = `<input type="text" class="prompt" placeholder="${escapeHtml(spec.prompt)}" value="${escapeHtml((r.values && r.values.prompt) || '')}">
        <button class="btn small primary">Generate</button>`;
      const inp = row.querySelector('input');
      const run = () => {
        const text = inp.value.trim();
        if (!text) { toast('Describe what should be there.'); inp.focus(); return; }
        inp.blur();
        sweep(r, spec.key, { ...(r.values || {}), prompt: text }, true);
      };
      row.querySelector('button').onclick = run;
      inp.onkeydown = (e) => { if (e.key === 'Enter') run(); };
      body.append(row);
    }
    spec.params.forEach((p, i) => body.append(buildParam(r, p, i === 0)));
    if (r.info) {
      const inf = document.createElement('div');
      inf.className = 'info';
      inf.textContent = r.info;
      body.append(inf);
    }
    if (spec.blend !== 'transparent') body.append(buildAdjSlider(r, { key: 'strength', label: 'Strength', min: 0, max: 1, step: 0.01, default: 1, unit: '',
      hint: 'Blend between the original (0) and the processed result (1).' }));
  }
  body.append(buildAdjustPanel(r));
  if (r.id !== 'bg' && !(r.blend === 'box' && r.restorer !== 'none')) {
    const edge = document.createElement('label');
    edge.className = 'edge';
    edge.innerHTML = `<div class="param-top"><span>Edge softness</span><span class="val">${fmt(r.feather, 0.1)} px</span></div>
      <input type="range" min="0" max="12" step="0.5" value="${r.feather}">`;
    const inp = edge.querySelector('input');
    inp.oninput = () => (edge.querySelector('.val').textContent = fmt(inp.value, 0.1) + ' px');
    inp.onchange = () => patchRegion(r, { feather: parseFloat(inp.value) });
    body.append(edge);
  }
  el.append(body);
  el.addEventListener('mousedown', (e) => {
    if (e.target.closest('input,select,button')) return;
    setActive(r.id, false);
  });
  el.addEventListener('mouseenter', () => { S.hoverRegion = r.id; drawOverlay(); });
  el.addEventListener('mouseleave', () => { S.hoverRegion = null; drawOverlay(); });
  return el;
}

function buildParam(r, p, primary) {
  const wrap = document.createElement('div');
  const v = r.values[p.key] ?? p.default;
  const est = r.estimate && r.estimate[p.key];
  wrap.innerHTML = `<div class="param-top"><span class="lbl">${escapeHtml(p.label)}</span><span class="val">${fmt(v, p.step)}</span></div>
    <div class="slider-wrap"><input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}"></div>`;
  const inp = wrap.querySelector('input');
  const val = wrap.querySelector('.val');
  if (p.key === 'hue') inp.classList.add('hue-slider');
  if (primary && (p.sweep.length || est !== undefined)) {
    const ticks = document.createElement('div');
    ticks.className = 'ticks';
    const pos = (x) => ((x - p.min) / (p.max - p.min)) * 100 + '%';
    for (const s of p.sweep) ticks.insertAdjacentHTML('beforeend', `<i style="left:${pos(s)}"></i>`);
    if (est !== undefined) ticks.insertAdjacentHTML('beforeend', `<b style="left:${pos(est)}" title="Model's own estimate">▲</b>`);
    wrap.querySelector('.slider-wrap').append(ticks);
  }
  let strip = null;
  if (primary && r.cands && r.cands.length > 1) {
    // every precomputed result as a thumbnail: click the one you like
    strip = document.createElement('div');
    strip.className = 'filmstrip';
    for (const c of r.cands) {
      const im = document.createElement('img');
      im.src = c.img.src;
      im.title = `${p.label} ${fmt(c.value, p.step)}`;
      im.dataset.v = c.value;
      im.onclick = () => { inp.value = c.value; inp.oninput(); inp.onchange(); };
      strip.append(im);
    }
    wrap.append(strip);
  }
  const markStrip = (x) => {
    if (!strip) return;
    let best = null;
    for (const im of strip.children) if (!best || Math.abs(im.dataset.v - x) < Math.abs(best.dataset.v - x)) best = im;
    for (const im of strip.children) im.classList.toggle('on', im === best);
  };
  markStrip(v);
  const hint = document.createElement('div');
  hint.className = 'param-hint';
  hint.textContent = p.hint + (est !== undefined ? `  ▲ = estimated (${fmt(est, p.step)})` : '');
  wrap.append(hint);
  inp.oninput = () => {
    const x = parseFloat(inp.value);
    val.textContent = fmt(x, p.step);
    r.values = { ...r.values, [p.key]: x };
    if (primary) { previewPrimary(r, x); markStrip(x); }
  };
  inp.onchange = () => {
    S.tuned = true; updateSteps();
    r.values = { ...r.values, [p.key]: parseFloat(inp.value) };
    // Changing a secondary control changes every sweep candidate.
    if (primary) renderExact(r); else sweep(r, r.restorer, r.values);
  };
  return wrap;
}

function fmtAdj(a, v) {
  if (a.key === 'strength') return Math.round(v * 100) + '%';
  return (v > 0 && a.key !== 'sharpen' ? '+' : '') + Number(v).toFixed(2) + (a.unit || '');
}

function buildAdjSlider(r, a) {
  const wrap = document.createElement('div');
  const v = adjValue(r, a.key);
  wrap.innerHTML = `<div class="param-top"><span class="lbl">${escapeHtml(a.label)}</span><span class="val">${fmtAdj(a, v)}</span></div>
    <input type="range" min="${a.min}" max="${a.max}" step="${a.step}" value="${v}" title="Double-click to reset">`;
  if (a.hint) wrap.insertAdjacentHTML('beforeend', `<div class="param-hint">${escapeHtml(a.hint)}</div>`);
  const inp = wrap.querySelector('input'), val = wrap.querySelector('.val');
  const set = (x) => {
    r.adjust = { ...(r.adjust || {}), [a.key]: x };
    val.textContent = fmtAdj(a, x);
    const badge = document.querySelector(`.card[data-id="${r.id}"] .adj-badge`);
    if (badge) badge.hidden = adjIdentity({ ...r, restorer: 'none' });
    scheduleComposite();
  };
  const commit = () => api(`/api/session/${sid()}/region/${r.id}`, { adjust: r.adjust }, 'PATCH').catch((e) => toast(e.message));
  inp.oninput = () => { S.draft = true; set(parseFloat(inp.value)); };
  inp.onchange = () => { S.draft = false; scheduleComposite(); commit(); };
  inp.ondblclick = () => { inp.value = a.default; set(a.default); commit(); };
  return wrap;
}

function buildAdjustPanel(r) {
  const d = document.createElement('details');
  d.className = 'adjust';
  d.open = !!r.adjOpen;
  d.ontoggle = () => (r.adjOpen = d.open);
  const changed = !adjIdentity({ ...r, restorer: 'none' });
  d.innerHTML = `<summary>Adjust <span class="adj-badge" ${changed ? '' : 'hidden'}>●</span></summary>`;
  const inner = document.createElement('div');
  inner.className = 'adjust-body';
  const commit = () => api(`/api/session/${sid()}/region/${r.id}`, { adjust: r.adjust }, 'PATCH').catch((e) => toast(e.message));
  const setMany = (vals, label) => {
    if (!Object.keys(vals).length) { toast('Nothing to correct here.'); return; }
    r.adjust = { ...(r.adjust || {}), ...vals };
    commit();
    r.adjOpen = true;
    renderCard(r);
    composite();
    toast(label, true);
  };
  const row = document.createElement('div');
  row.className = 'adj-actions';
  const mk = (text, title, fn) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; b.title = title; b.onclick = fn; row.append(b); };
  mk('Auto tone', 'Set exposure, whites and blacks from this region\'s histogram', () => setMany(autoTone(r), 'Auto tone applied'));
  mk('Auto color', 'Neutralise the colour cast (gray-world white balance)', () => setMany(autoColor(r), 'Auto color applied'));
  mk('Reset', 'Reset all adjustments (keeps Strength)', () => {
    r.adjust = r.adjust && r.adjust.strength !== undefined ? { strength: r.adjust.strength } : {};
    commit(); r.adjOpen = true; renderCard(r); composite();
  });
  inner.append(row);
  let group = null;
  for (const a of S.adjSpec) {
    if (a.group !== group) {
      group = a.group;
      const hdr = document.createElement('div');
      hdr.className = 'adj-group';
      hdr.textContent = group;
      inner.append(hdr);
    }
    inner.append(buildAdjSlider(r, a));
  }
  d.append(inner);
  return d;
}

function iconBtn(svg, title, fn, off = false) {
  const b = document.createElement('button');
  b.className = 'icon-btn' + (off ? ' off' : '');
  b.title = title;
  b.innerHTML = svg;
  b.onclick = (e) => { e.stopPropagation(); fn(); };
  return b;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function patchRegion(r, patch) {
  const info = await busy('Updating…', () => api(`/api/session/${sid()}/region/${r.id}`, patch, 'PATCH'));
  await upsertRegion(info);
  if (patch.name) { if (S.active === r.id) $('#editing-name').textContent = r.name; }
  else renderCard(r);
  composite();
}

async function toggleVisible(r) {
  r.visible = !r.visible;
  renderCard(r);
  composite();
  api(`/api/session/${sid()}/region/${r.id}`, { visible: r.visible }, 'PATCH').then(refreshBackground).catch((e) => toast(e.message));
}

async function deleteRegion(r) {
  await busy('Deleting…', () => api(`/api/session/${sid()}/region/${r.id}`, undefined, 'DELETE'));
  S.regions = S.regions.filter((x) => x !== r);
  if (S.active === r.id || S.editing === r.id) doneEditing();
  renderPanel();
  composite();
  await refreshBackground();
}

async function moveRegion(r, dir) {
  const i = S.regions.indexOf(r), j = i + dir;
  if (j < 1 || j >= S.regions.length) return;
  [S.regions[i], S.regions[j]] = [S.regions[j], S.regions[i]];
  await api(`/api/session/${sid()}/order`, { order: S.regions.slice(1).map((x) => x.id) }).catch((e) => toast(e.message));
  renderPanel();
  composite();
}

// ---------------------------------------------------------------- events
viewport.addEventListener('contextmenu', (e) => e.preventDefault());
$('#help').onclick = () => ($('#help').hidden = true);

viewport.addEventListener('pointerdown', (e) => {
  // Floating controls live inside the viewport; let them handle their own clicks.
  if (!S.orig || e.target.closest('.toolbar, .editing, .empty, .cropbar, .modehint')) return;
  const p = toImage(e);
  if (S.crop) {
    const mode = e.button === 0 ? cropHit(p) : null;
    if (mode) { viewport.setPointerCapture(e.pointerId); S.crop.drag = { mode, sx: p.sx, sy: p.sy, rect0: [...S.crop.rect] }; }
    return;
  }
  viewport.setPointerCapture(e.pointerId);
  if (e.button === 0 && nearSplit(p)) { S.drag = { type: 'split' }; return; }
  if (e.button === 1 || (e.button === 0 && S.tool === 'pan')) {
    S.drag = { type: 'pan', sx: p.sx, sy: p.sy, vx: S.view.x, vy: S.view.y };
    viewport.classList.add('panning');
    return;
  }
  if (e.button === 0 && S.tool === 'box') { S.drag = { type: 'box', x0: p.sx, y0: p.sy, x1: p.sx, y1: p.sy }; return; }
  if ((e.button === 0 || e.button === 2) && S.tool === 'brush') {
    S.drag = { type: 'brush', pts: [[p.ix, p.iy]], radius: S.brush / S.view.k, add: e.button === 0 && !e.altKey };
    drawOverlay();
    return;
  }
  if (e.button === 0 || e.button === 2) S.drag = { type: 'click', sx: p.sx, sy: p.sy, negative: e.button === 2 || e.altKey };
});

viewport.addEventListener('pointermove', (e) => {
  if (!S.orig) return;
  const p = toImage(e);
  if (S.crop) {
    const cd = S.crop.drag;
    if (cd) { cropDrag(cd, p); drawOverlay(); return; }
    const m = cropHit(p);
    viewport.style.cursor = !m ? 'default' : m === 'move' ? 'move' : ({ n: 'ns', s: 'ns', e: 'ew', w: 'ew', nw: 'nwse', se: 'nwse', ne: 'nesw', sw: 'nesw' })[m] + '-resize';
    return;
  }
  const d = S.drag;
  viewport.style.cursor = !d && nearSplit(p) ? 'ew-resize' : '';
  if (S.tool === 'brush') { S.hover = p; if (!d) drawOverlay(); }
  if (!d) return;
  if (d.type === 'brush') {
    const [lx, ly] = d.pts[d.pts.length - 1];
    if (Math.hypot(p.ix - lx, p.iy - ly) > d.radius / 3) d.pts.push([p.ix, p.iy]);
    drawOverlay();
    return;
  }
  if (d.type === 'pan') { S.view.x = d.vx + p.sx - d.sx; S.view.y = d.vy + p.sy - d.sy; applyView(); }
  else if (d.type === 'box') { d.x1 = p.sx; d.y1 = p.sy; drawOverlay(); }
  else if (d.type === 'split') { S.split = Math.max(0, Math.min(1, p.ix / canvas.width)); composite(); }
  else if (d.type === 'click' && Math.hypot(p.sx - d.sx, p.sy - d.sy) > 5) {
    // Dragging from a click turns into a pan, which is what people expect.
    S.drag = { type: 'pan', sx: d.sx, sy: d.sy, vx: S.view.x, vy: S.view.y };
    viewport.classList.add('panning');
  }
});

viewport.addEventListener('pointerup', (e) => {
  if (S.crop) { S.crop.drag = null; return; }
  const d = S.drag;
  S.drag = null;
  viewport.classList.remove('panning');
  if (!d || !S.orig) return;
  const p = toImage(e);
  if (d.type === 'click' && inImage(p)) clickPrompt(p, d.negative).catch(() => {});
  else if (d.type === 'click' && !d.negative) doneEditing();   // click beside the image: deselect
  if (d.type === 'brush') brushStroke(d).catch(() => {}).finally(drawOverlay);
  if (d.type === 'box') {
    const a = { ix: (Math.min(d.x0, d.x1) - S.view.x) / S.view.k, iy: (Math.min(d.y0, d.y1) - S.view.y) / S.view.k };
    const b = { ix: (Math.max(d.x0, d.x1) - S.view.x) / S.view.k, iy: (Math.max(d.y0, d.y1) - S.view.y) / S.view.k };
    const clamp = (v, m) => Math.max(0, Math.min(m, v));
    const box = [clamp(a.ix, canvas.width), clamp(a.iy, canvas.height), clamp(b.ix, canvas.width), clamp(b.iy, canvas.height)];
    drawOverlay();
    if (box[2] - box[0] > 4 && box[3] - box[1] > 4) boxPrompt(box).catch(() => {});
  }
});

viewport.addEventListener('pointerleave', () => { S.hover = null; drawOverlay(); });

viewport.addEventListener('wheel', (e) => {
  if (!S.orig || S.crop) return;
  e.preventDefault();
  const p = toImage(e);
  zoomAt(p.sx, p.sy, S.view.k * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
}, { passive: false });

const MODE_HINTS = {
  point: 'Select: <b>click</b> to add · <b>right-click</b> to remove · drag to pan · wheel to zoom',
  box: 'Box: <b>drag</b> a box around something · <kbd>V</kbd> back to click-select',
  brush: 'Brush: <b>paint</b> to add · <b>right-drag</b> to erase · <kbd>[</kbd> <kbd>]</kbd> size · <kbd>V</kbd> back to click-select',
  pan: 'Pan: drag to move · wheel to zoom',
  crop: '',
};

function setTool(t) {
  if (t === 'crop' && !S.crop) enterCrop();
  if (t !== 'crop' && S.crop) exitCrop();
  S.tool = t;
  $('#brush-size').hidden = t !== 'brush';
  document.querySelectorAll('.tool[data-tool]').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  viewport.className = 'viewport tool-' + t;
  const mh = $('#modehint');
  mh.hidden = !S.orig || !MODE_HINTS[t];
  mh.classList.toggle('attn', t === 'pan');
  mh.innerHTML = MODE_HINTS[t] + (t === 'pan' ? ' <button class="btn small" id="mh-back">Back to selecting (V)</button>' : '');
  const back = $('#mh-back');
  if (back) back.onclick = () => setTool('point');
}
document.querySelectorAll('.tool[data-tool]').forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
setTool('point');

$('#zoom-fit').onclick = fit;
$('#zoom-1').onclick = () => zoomAt(viewport.clientWidth / 2, viewport.clientHeight / 2, 1 / (S.sess ? S.sess.scale : 1));
$('#btn-done').onclick = doneEditing;
$('#btn-del-region').onclick = () => { const r = region(S.editing || S.active); if (r && r.id !== 'bg') deleteRegion(r).catch(() => {}); };
$('#btn-home').onclick = goHome;
$('#crop-ratio').onchange = cropChanged;
$('#crop-rotl').onclick = () => { S.crop.rot = (S.crop.rot + 3) % 4; cropChanged(); };
$('#crop-rotr').onclick = () => { S.crop.rot = (S.crop.rot + 1) % 4; cropChanged(); };
$('#crop-fliph').onclick = () => { S.crop.flipH = !S.crop.flipH; drawOverlay(); };
$('#crop-flipv').onclick = () => { S.crop.flipV = !S.crop.flipV; drawOverlay(); };
$('#crop-angle').oninput = (e) => { S.crop.angle = parseFloat(e.target.value); cropChanged(); };
$('#crop-angle').ondblclick = (e) => { e.target.value = 0; S.crop.angle = 0; cropChanged(); };
$('#crop-reset').onclick = () => { S.crop.rot = 0; S.crop.flipH = S.crop.flipV = false; S.crop.angle = 0; $('#crop-angle').value = 0; $('#crop-ratio').value = 'free'; cropChanged(); };
$('#crop-cancel').onclick = () => setTool('point');
$('#crop-apply').onclick = () => applyCrop().catch(() => {});
$('#btn-undo').onclick = () => historyStep('undo').catch(() => {});
$('#btn-undo-all').onclick = () => historyStep('undo').catch(() => {});
$('#btn-redo-all').onclick = () => historyStep('redo').catch(() => {});
$('#brush-size input').oninput = (e) => { S.brush = parseFloat(e.target.value); drawOverlay(); };

function setCompare(on) {
  if (S.compare === on) return;
  S.compare = on;
  $('#btn-compare').classList.toggle('active', on);
  composite();
}
const cmp = $('#btn-compare');
cmp.addEventListener('pointerdown', () => setCompare(true));
for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) cmp.addEventListener(ev, () => setCompare(false));

$('#btn-auto').onclick = () => autoEnhance().catch(() => {});
$('#steps-hide').onclick = () => { try { localStorage.setItem(STEPS_KEY, '1'); } catch (_) {} updateSteps(); };
window.addEventListener('beforeunload', (e) => {
  if (S.dirty) { e.preventDefault(); e.returnValue = ''; }   // unsaved edits: ask before leaving
});
$('#btn-subject').onclick = () => selectSubject().catch(() => {});
$('#btn-removebg').onclick = () => removeBackground().catch(() => {});
$('#btn-apply').onclick = async () => {
  if (S.regions.every((r) => r.restorer === 'none' && adjIdentity(r))) { toast('Nothing to apply yet.'); return; }
  const info = await busy('Applying (rendering full resolution)…', () => api(`/api/session/${sid()}/apply`, {}));
  await openSession(info, true);
  toast('Applied. The result is now the image you are editing (Ctrl+Z to undo).', true);
};


$('#btn-split').onclick = () => {
  S.split = S.split === null ? 0.5 : null;
  $('#btn-split').classList.toggle('active', S.split !== null);
  composite();
};

const EXPORT_KEY = 'rap.export';
function exportPrefs() {
  try { return JSON.parse(localStorage.getItem(EXPORT_KEY)) || {}; } catch (_) { return {}; }
}
function exportUpdate() {
  const fmt = $('#export-fmt .on').dataset.v, size = $('#export-size').value;
  $('#export-q-row').hidden = fmt === 'png';
  $('#export-q-val').textContent = $('#export-q').value;
  if (!S.sess) return;
  let w = S.sess.width, h = S.sess.height;
  if (size.startsWith('m')) { const m = parseInt(size.slice(1)), k = Math.min(1, m / Math.max(w, h)); w = Math.round(w * k); h = Math.round(h * k); }
  else { const k = Math.min(parseFloat(size), 8192 / Math.max(w, h)); w = Math.round(w * k); h = Math.round(h * k); }
  $('#export-info').textContent = `${w} × ${h} px` + (parseFloat(size) > 1 ? ' · upscaling takes a while' : '')
    + (fmt === 'jpeg' && isTransparent() ? ' · JPEG has no transparency (white background)' : '');
  try { localStorage.setItem(EXPORT_KEY, JSON.stringify({ fmt, size, q: $('#export-q').value, meta: $('#export-meta').checked })); } catch (_) {}
}
(function initExport() {
  const p = exportPrefs();
  if (p.fmt) document.querySelectorAll('#export-fmt button').forEach((b) => b.classList.toggle('on', b.dataset.v === p.fmt));
  if (p.size) $('#export-size').value = p.size;
  if (p.q) $('#export-q').value = p.q;
  if (p.meta !== undefined) $('#export-meta').checked = p.meta;
  document.querySelectorAll('#export-fmt button').forEach((b) => (b.onclick = () => {
    document.querySelectorAll('#export-fmt button').forEach((x) => x.classList.toggle('on', x === b)); exportUpdate();
  }));
  for (const id of ['#export-size', '#export-q', '#export-meta']) $(id).oninput = $(id).onchange = exportUpdate;
  exportUpdate();
})();
$('#btn-export').onclick = (e) => { e.stopPropagation(); $('#export-panel').hidden = !$('#export-panel').hidden; exportUpdate(); };
document.addEventListener('mousedown', (e) => { if (!e.target.closest('.export-wrap')) $('#export-panel').hidden = true; });
$('#export-go').onclick = async () => {
  const fmt = $('#export-fmt .on').dataset.v, size = $('#export-size').value;
  const q = new URLSearchParams({ fmt, quality: $('#export-q').value, meta: $('#export-meta').checked });
  if (size.startsWith('m')) q.set('max_side', size.slice(1)); else q.set('scale', size);
  $('#export-panel').hidden = true;
  const up = parseFloat(size) > 1;
  await busy(up ? `Rendering and upscaling ×${size} (this can take a while)…` : 'Rendering full resolution…', async () => {
    const r = await fetch(`/api/session/${sid()}/export?${q}`);
    if (!r.ok) throw new Error('export failed');
    const got = parseFloat(r.headers.get('X-Export-Scale') || '1');
    if (up && got < parseFloat(size)) toast(`Upscaled ×${got} instead of ×${size} to keep the output under 8192 px.`, true);
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = r.headers.get('X-Export-Name') || 'restored.png';
    a.click();
    S.dirty = false; S.exported = true; updateSteps();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  });
};

for (const id of ['#file', '#file2']) $(id).onchange = (e) => { upload(e.target.files[0]).catch(() => {}); e.target.value = ''; };

$('#find').onsubmit = (e) => {
  e.preventDefault();
  const q = $('#find-input').value.trim();
  $('#find-input').blur();  // hand the keyboard back to the canvas shortcuts
  if (q && S.sess) findText(q, $('#find-separate').checked).catch(() => {});
};

window.addEventListener('paste', (e) => {
  if (typing(e)) return;
  const f = [...(e.clipboardData ? e.clipboardData.files : [])].find((x) => x.type.startsWith('image/'));
  if (f) { e.preventDefault(); upload(f).catch(() => {}); }
});

viewport.addEventListener('dragover', (e) => { e.preventDefault(); viewport.classList.add('dragover'); });
viewport.addEventListener('dragleave', () => viewport.classList.remove('dragover'));
viewport.addEventListener('drop', (e) => {
  e.preventDefault();
  viewport.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) upload(f).catch(() => {});
});

// Only text fields swallow shortcuts; sliders and dropdowns should not block Space-to-compare.
const typing = (e) => e.target.closest('input[type=text], textarea');
window.addEventListener('keydown', (e) => {
  if (typing(e)) return;
  if (e.key === '?' || (e.key === 'Escape' && !$('#help').hidden)) { $('#help').hidden = !$('#help').hidden || e.key === 'Escape'; return; }
  if (!S.orig) return;
  if (S.crop) {
    if (e.key === 'Escape') setTool('point');
    else if (e.key === 'Enter') applyCrop().catch(() => {});
    return;
  }
  if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) setCompare(true); }
  else if (e.key === 'Escape' || e.key === 'Enter') doneEditing();
  else if (e.key === 'f' || e.key === 'F') fit();
  else if (e.key === '1') $('#zoom-1').click();
  else if (e.key === 'v' || e.key === 'V') setTool('point');
  else if (e.key === 'b' || e.key === 'B') setTool('box');
  else if (e.key === 'h' || e.key === 'H') setTool('pan');
  else if (e.key === 'p' || e.key === 'P') setTool('brush');
  else if (e.key === 'c' || e.key === 'C') setTool('crop');
  else if (e.key === '[') { S.brush = Math.max(4, S.brush / 1.25); $('#brush-size input').value = S.brush; drawOverlay(); }
  else if (e.key === ']') { S.brush = Math.min(120, S.brush * 1.25); $('#brush-size input').value = S.brush; drawOverlay(); }
  else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); historyStep(e.shiftKey ? 'redo' : 'undo').catch(() => {}); }
  else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); historyStep('redo').catch(() => {}); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && S.active && S.active !== 'bg') deleteRegion(region(S.active));
});
window.addEventListener('keyup', (e) => {
  if (e.code !== 'Space' || typing(e)) return;
  e.preventDefault();  // otherwise Space would also press a focused button
  setCompare(false);
});
window.addEventListener('resize', () => { if (S.orig) drawOverlay(); });

// ------------------------------------------------------------------ init
$('#models-chip').onclick = (e) => { e.stopPropagation(); $('#models-panel').hidden = !$('#models-panel').hidden; refreshModels(); };
document.addEventListener('mousedown', (e) => {
  if (!e.target.closest('.models-wrap')) $('#models-panel').hidden = true;
  if (!e.target.closest('#picker, .pick-btn')) closePicker();
});
window.addEventListener('resize', closePicker);
$('#regions').addEventListener('scroll', closePicker);
$('#models-free').onclick = async () => {
  await busy('Freeing GPU memory…', () => api('/api/models/free', {}));
  refreshModels();
  toast('Models moved to CPU memory; they come back in seconds when used.', true);
};

(async function init() {
  pollModels();
  const cfg = await api('/api/config');
  for (const r of cfg.restorers) { S.restorers[r.key] = r; S.order.push(r.key); }
  S.adjSpec = cfg.adjustments || [];
  const ex = $('#examples');
  for (const e of cfg.examples) {
    const b = document.createElement('button');
    b.title = e.file;
    b.innerHTML = `<img src="/api/example/${encodeURIComponent(e.file)}?thumb=1" alt="" loading="lazy">
      <span class="ex-text"><span class="ex-title">${escapeHtml(e.title)}</span>
      <span class="ex-hint">${escapeHtml(e.hint)}</span>
      ${e.credit ? `<span class="ex-credit">${escapeHtml(e.credit)}</span>` : ''}</span>`;
    b.onclick = () => openExample(e.file).catch(() => {});
    ex.append(b);
  }
  await reattach();
})().catch((e) => {
  toast('Could not reach the server: ' + e.message);
  fetch('/api/client-error', { method: 'POST', body: 'init: ' + e.message + ' · ' + navigator.userAgent }).catch(() => {});
});
