# --------------------------------------------
# Model manager: lazy loading and GPU memory
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Load models when they are first needed and keep GPU memory within a budget.

Every model is registered with a factory. `get(key)` builds it on first use,
moves it back from CPU memory if it was offloaded, and records the use. When
the models on the GPU would exceed the budget, the least recently used ones
are moved to CPU memory (seconds to bring back, instead of reloading from
disk). Models idle for a while are offloaded too, which is friendlier on a
shared GPU.

Eviction only happens on the inference path (the caller holds the GPU lock),
never from the background preloader, so a model is never moved while in use.

The budget follows the GPU: by default about 60% of its memory holds model
weights, the rest is left for the work itself. Models parked in CPU memory are
unloaded (least recently used first) when they would take more than half of
the system RAM. Very large pipelines can manage their own memory by keeping
their components in CPU memory between uses (`self_offloading`); the manager
then leaves them alone.

Running out of GPU memory (loading a model or running it) moves every other
model to CPU memory and tries once more; if that fails too, the user gets a
clear message instead of a crash.
"""
import gc
import os
import threading
import time

import torch

from .restorers.base import Restorer


def _torch_attrs(obj):
    """Modules, tensors and diffusers pipelines held directly by obj."""
    out = []
    for name, v in vars(obj).items():
        if isinstance(v, (torch.nn.Module, torch.Tensor)) or (hasattr(v, 'components') and hasattr(v, 'to')):
            out.append((name, v))
    return out


def _module_bytes(m):
    return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))


def gpu_bytes(obj):
    n = 0
    for _, v in _torch_attrs(obj):
        if isinstance(v, torch.Tensor):
            n += v.numel() * v.element_size()
        elif isinstance(v, torch.nn.Module):
            n += _module_bytes(v)
        else:
            n += sum(_module_bytes(c) for c in v.components.values() if isinstance(c, torch.nn.Module))
    extra = getattr(obj, 'extra_modules', None)
    return n + (sum(_module_bytes(m) for m in extra()) if extra else 0)


def move(obj, device):
    for name, v in _torch_attrs(obj):
        if isinstance(v, torch.Tensor):
            setattr(obj, name, v.to(device))
        elif isinstance(v, torch.nn.Module):
            v.to(device)
        else:  # diffusers pipeline: move its modules (pipe.to('cpu') warns for fp16 pipelines)
            for c in v.components.values():
                if isinstance(c, torch.nn.Module):
                    c.to(device)
    for m in (obj.extra_modules() if hasattr(obj, 'extra_modules') else []):
        m.to(device)


GPU_GB = None   # GPU memory this process may use; restorers size themselves by it


def gpu_gb():
    return GPU_GB or torch.cuda.get_device_properties(0).total_memory / 1e9


def _system_ram_gb():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    except (ValueError, OSError, AttributeError):
        return 32.0


def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in str(err).lower()


class Entry:
    def __init__(self, key, label, factory, pinned=False):
        self.key, self.label, self.factory, self.pinned = key, label, factory, pinned
        self.obj, self.where, self.size, self.last = None, 'unloaded', 0, 0.0
        self.error = None
        self.self_off = False            # the model keeps its own components in CPU memory between uses
        self.lock = threading.Lock()     # one load / move at a time per model


class ModelManager:
    def __init__(self, device, budget_gb=None, idle_minutes=20, log=print, gpu_limit_gb=None, cpu_budget_gb=None):
        global GPU_GB
        GPU_GB = gpu_limit_gb or torch.cuda.get_device_properties(0).total_memory / 1e9
        self.device = device
        self.budget = (budget_gb or min(32.0, 0.6 * GPU_GB)) * 1e9
        self.cpu_budget = (cpu_budget_gb or 0.5 * _system_ram_gb()) * 1e9
        log(f'GPU memory: {GPU_GB:.0f} GB usable, up to {self.budget / 1e9:.0f} GB of it for model weights; '
            f'up to {self.cpu_budget / 1e9:.0f} GB of models parked in CPU memory')
        self.idle = idle_minutes * 60
        self.log = log
        self.entries = {}
        self.lock = threading.RLock()
        self.cache_droppers = []          # callables releasing recomputable GPU caches (see free_gpu)

    # ---- registration --------------------------------------------------
    def register(self, key, label, factory):
        self.entries[key] = Entry(key, label, factory)

    def add_pinned(self, key, label, obj):
        """An already-loaded model that always stays on the GPU (counts toward the budget)."""
        e = Entry(key, label, None, pinned=True)
        e.obj, e.where, e.size, e.last = obj, 'cuda', gpu_bytes(obj), time.time()
        self.entries[key] = e

    # ---- use -----------------------------------------------------------
    def used(self):
        return sum(e.size for e in self.entries.values() if e.where == 'cuda' and not e.self_off)

    def cpu_used(self):
        return sum(e.size for e in self.entries.values() if e.where == 'cpu' or (e.self_off and e.obj is not None))

    @staticmethod
    def _movable(e):
        return e.where == 'cuda' and not e.pinned and not e.self_off

    def _trim_cpu(self, keep=None):
        """Unload the least recently used parked models when they would fill the CPU memory."""
        with self.lock:
            parked = sorted((e for e in self.entries.values()
                             if e is not keep and e.obj is not None and e.factory is not None
                             and (e.where == 'cpu' or e.self_off)), key=lambda e: e.last)
            unloaded = False
            while self.cpu_used() > self.cpu_budget and parked:
                v = parked.pop(0)
                with v.lock:
                    v.obj, v.where, v.size, v.self_off = None, 'unloaded', 0, False
                unloaded = True
                self.log(f'unloaded {v.label} to keep CPU memory free (it reloads from disk when needed)')
            if unloaded:
                gc.collect()

    def _make_room(self, need, keep):
        with self.lock:
            victims = sorted((e for e in self.entries.values()
                              if self._movable(e) and e is not keep), key=lambda e: e.last)
            freed = False
            while self.used() + need > self.budget and victims:
                v = victims.pop(0)
                with v.lock:
                    move(v.obj, 'cpu')
                    v.where = 'cpu'
                freed = True
                self.log(f'offloaded {v.label} to CPU memory to stay within the GPU budget')
            if freed:
                torch.cuda.empty_cache()
                self._trim_cpu(keep)

    def free_gpu(self, keep=None):
        """After running out of GPU memory: move every other model to CPU memory and drop the
        caches that can be recomputed (prepared crops, SAM features of other images)."""
        with self.lock:
            for e in self.entries.values():
                if self._movable(e) and e is not keep:
                    with e.lock:
                        move(e.obj, 'cpu')
                        e.where = 'cpu'
        for drop in self.cache_droppers:
            drop()
        gc.collect()
        torch.cuda.empty_cache()
        self._trim_cpu(keep)
        self.log(f'  {torch.cuda.memory_allocated() / 1e9:.1f} GB still in use on the GPU')

    def _build(self, e):
        for attempt in (0, 1):
            try:
                obj = e.factory()
                if hasattr(obj, 'warm'):
                    obj.warm()
                return obj
            except Exception as err:
                obj = None
                if attempt or not is_oom(err):
                    raise
            self.log(f'out of GPU memory loading {e.label}: moving the other models to CPU memory and retrying')
            self.free_gpu(keep=e)

    def _to_gpu(self, e):
        for attempt in (0, 1):
            try:
                move(e.obj, self.device)
                return
            except Exception as err:
                if not is_oom(err):
                    raise
            if attempt == 0:
                self.free_gpu(keep=e)
        # outside the except block, so the failed attempt's tensors can be freed
        move(e.obj, 'cpu')
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f'Not enough GPU memory for {e.label}.')

    def get(self, key, evict=True):
        e = self.entries[key]
        with e.lock:
            if e.obj is None:
                if evict:
                    self._make_room(0, e)
                e.where = 'loading'
                t = time.time()
                oom = False
                try:
                    e.obj = self._build(e)
                except Exception as err:
                    e.obj = None
                    if not is_oom(err):
                        e.where, e.error = 'failed', f'{type(err).__name__}: {err}'
                        raise RuntimeError(f'{e.label} could not be loaded: {e.error}') from err
                    oom = True
                if oom:   # handled outside the except block, so the half-built model can be freed
                    e.where = 'unloaded'
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(f'Not enough GPU memory to load {e.label}. '
                                       'Close other GPU programs or use a larger GPU.')
                e.size, e.where = gpu_bytes(e.obj), 'cuda'
                e.self_off = bool(getattr(e.obj, 'self_offloading', False))
                how = ', kept in CPU memory between uses' if e.self_off else ''
                self.log(f'loaded {e.label} ({e.size / 1e9:.1f} GB{how}) in {time.time() - t:.0f} s')
            elif e.where == 'cpu':
                if evict:
                    self._make_room(e.size, e)
                self._to_gpu(e)
                e.where = 'cuda'
            e.last = time.time()
        if evict and self.used() > self.budget:
            self._make_room(0, e)
        if evict:
            self._trim_cpu(keep=e)
        return e.obj

    def state(self, key):
        return self.entries[key].where if key in self.entries else 'cuda'

    def status(self):
        return {'budget_gb': self.budget / 1e9, 'used_gb': self.used() / 1e9, 'usable_gb': GPU_GB,
                'models': [{'key': e.key, 'label': e.label, 'state': e.where, 'pinned': e.pinned,
                            'size_gb': round(e.size / 1e9, 2), 'error': e.error} for e in self.entries.values()]}

    def offload_all(self, gpu_lock):
        with gpu_lock:
            for e in self.entries.values():
                if self._movable(e):
                    with e.lock:
                        move(e.obj, 'cpu')
                        e.where = 'cpu'
            torch.cuda.empty_cache()
            self._trim_cpu()

    # ---- background work -----------------------------------------------
    def preload(self, keys):
        """Warm up the given models in a background thread, only while they fit."""
        def run():
            for k in keys:
                e = self.entries.get(k)
                if e is None or e.obj is not None:
                    continue
                if self.used() + 5e9 > self.budget:   # no evictions for a mere warm-up
                    continue
                try:
                    self.log(f'warming up {e.label} in the background')
                    self.get(k, evict=False)
                except Exception as err:
                    self.log(f'  {err}')
        threading.Thread(target=run, daemon=True, name='preload').start()

    def start_idle_offload(self, gpu_lock):
        def run():
            while True:
                time.sleep(60)
                now = time.time()
                idle = [e for e in self.entries.values() if self._movable(e) and now - e.last > self.idle]
                if not idle or not gpu_lock.acquire(blocking=False):
                    continue
                try:
                    for e in idle:
                        with e.lock:
                            move(e.obj, 'cpu')
                            e.where = 'cpu'
                        self.log(f'{e.label} idle for {self.idle // 60} min: moved to CPU memory')
                    torch.cuda.empty_cache()
                    self._trim_cpu()
                finally:
                    gpu_lock.release()
        if self.idle > 0:
            threading.Thread(target=run, daemon=True, name='idle-offload').start()


def _oom_safe(fn, name, obj, label, manager, entry):
    """Wrap a restorer method so running out of GPU memory degrades instead of failing:
    first the other models move to CPU memory, then a sweep (`render_many`) is split into
    smaller batches, then the restorer may shrink its own tiles/batches (`shrink()`)."""
    def attempt(*args, **kwargs):
        try:
            return True, fn(*args, **kwargs)
        except Exception as err:
            if not is_oom(err):
                raise
        gc.collect()      # outside the except block: the failed attempt's tensors are released
        torch.cuda.empty_cache()
        return False, None

    def call(*args, **kwargs):
        chunk = obj.__dict__.get('_sweep_chunk')   # a sweep size known to fit, learned from a split
        if name == 'render_many' and chunk and len(args) >= 2 and len(args[1]) > chunk:
            state, values = args[0], list(args[1])
            return [o for i in range(0, len(values), chunk) for o in call(state, values[i:i + chunk], *args[2:], **kwargs)]
        ok, out = attempt(*args, **kwargs)
        if ok:
            return out
        manager.log(f'out of GPU memory in {label}: moving the other models to CPU memory and retrying')
        manager.free_gpu(keep=entry)
        while True:
            ok, out = attempt(*args, **kwargs)
            if ok:
                return out
            if name == 'render_many' and len(args) >= 2 and len(args[1]) > 1:
                state, values = args[0], list(args[1])
                half = len(values) // 2
                obj.__dict__['_sweep_chunk'] = half
                manager.log(f'{label}: rendering {len(values)} settings in two halves to fit in GPU memory')
                return call(state, values[:half], *args[2:], **kwargs) + call(state, values[half:], *args[2:], **kwargs)
            if not (hasattr(obj, 'shrink') and obj.shrink()):
                break
            manager.log(f'{label}: working in smaller pieces to fit in GPU memory')
        raise RuntimeError(f'Not enough GPU memory to run {label} on this region. '
                           'Try a smaller selection or a smaller photo.')
    return call


class LazyRestorer:
    """Stands in for a restorer: metadata (key, label, params, flags, spec) comes
    from the class without loading anything; methods load it on first use."""

    def __init__(self, cls, manager):
        self._cls, self._m = cls, manager

    def __getattr__(self, name):
        cls = self.__dict__['_cls']
        if not hasattr(cls, name):
            raise AttributeError(name)
        v = getattr(cls, name)
        if not callable(v) or isinstance(v, type):
            return v
        m = self.__dict__['_m']
        obj = m.get(cls.key)
        return _oom_safe(getattr(obj, name), name, obj, cls.label, m, m.entries[cls.key])

    def spec(self):
        return Restorer.spec(self)

    def defaults(self):
        return self._cls.defaults(self)

    @property
    def state(self):
        return self._m.state(self._cls.key)
