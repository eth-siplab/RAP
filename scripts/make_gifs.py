# --------------------------------------------
# README demo GIFs
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

"""Turn the clips from record_demos.py into README GIFs (needs ffmpeg).

Model waits are sped up (each squeezed to at most MAX_WAIT seconds), the rest
plays at SPEED. Output: docs/demos/<name>.gif and docs/videos/<name>.mp4

    python scripts/make_gifs.py [name ...]
"""
import glob
import json
import os
import subprocess
import sys

OUT = 'docs/demos'
VIDEOS = 'docs/videos'
SPEED = 1.15       # overall speed-up of the interaction
MAX_WAIT = 0.7     # seconds a model wait may take in the GIF
FPS = 10
WIDTH = 800


def segments(meta, speed=SPEED):
    t, segs = meta['start'], []
    for a, b in sorted(meta['fast']):
        a, b = max(a, t), min(b, meta['end'])
        if b <= a:
            continue
        if a > t:
            segs.append((t, a, speed))
        segs.append((a, b, max(speed, (b - a) / MAX_WAIT)))
        t = b
    segs.append((t, meta['end'], speed))
    return [s for s in segs if s[1] - s[0] > 0.05]


def make(name):
    meta = json.load(open(f'{OUT}/{name}.json'))
    segs = segments(meta, {'hero': 1.5}.get(name, SPEED))   # the teaser moves a little faster
    fps = {'hero': 8}.get(name, FPS)
    width = {'hero': 760}.get(name, WIDTH)
    parts = [f'[0:v]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{sp:.3f}[v{i}]' for i, (a, b, sp) in enumerate(segs)]
    chain = ';'.join(parts) + ';' + ''.join(f'[v{i}]' for i in range(len(segs))) + f'concat=n={len(segs)}:v=1:a=0[c];' \
        f'[c]fps={fps},scale={width}:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=128:stats_mode=diff[p];' \
        f'[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle'
    gif = f'{OUT}/{name}.gif'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-threads', '2', '-i', f'{OUT}/{name}.webm',
                    '-filter_complex', chain, gif], check=True)
    dur = sum((b - a) / sp for a, b, sp in segs)
    print(f'{name}: {dur:.1f} s, {os.path.getsize(gif) / 2**20:.1f} MB')
    # the project page plays the same cut as an MP4: sharper and smaller than the GIF
    os.makedirs(VIDEOS, exist_ok=True)
    mp4 = f'{VIDEOS}/{name}.mp4'
    vchain = ';'.join(parts) + ';' + ''.join(f'[v{i}]' for i in range(len(segs))) + \
        f'concat=n={len(segs)}:v=1:a=0,fps=30,scale=1280:-2:flags=lanczos,format=yuv420p[o]'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-threads', '4', '-i', f'{OUT}/{name}.webm',
                    '-filter_complex', vchain, '-map', '[o]', '-c:v', 'libx264', '-preset', 'slow', '-crf', '26',
                    '-movflags', '+faststart', '-an', mp4], check=True)
    print(f'{name}: {os.path.getsize(mp4) / 2**20:.1f} MB mp4')


if __name__ == '__main__':
    names = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob(f'{OUT}/*.json'))]
    for n in names:
        make(n)
