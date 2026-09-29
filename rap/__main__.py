# --------------------------------------------
# RAP command line (python -m rap)
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import argparse
import os
import socket


def main():
    ap = argparse.ArgumentParser(prog='python -m rap', description='Restore Anything Pipeline (RAP): interactive region-wise image restoration')
    ap.add_argument('--gpu', default=None, help='GPU index to use (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights'))
    ap.add_argument('--cpu-threads', type=int, default=8,
                    help='CPU threads for torch/OpenCV. Their default (all cores) is very slow on busy shared machines.')
    ap.add_argument('--gpu-budget', type=float, default=None,
                    help='GB of GPU memory for model weights (default: 60%% of the GPU, at most 32); '
                         'least recently used models move to CPU memory beyond it.')
    ap.add_argument('--max-gpu-memory', type=float, default=0,
                    help='GB of GPU memory this process may use in total (0 = no limit), e.g. on a shared GPU.')
    ap.add_argument('--idle-offload', type=float, default=20,
                    help='minutes after which an unused model moves to CPU memory (0 = never).')
    ap.add_argument('--session-dir', default=os.path.expanduser('~/.cache/rap/sessions'),
                    help='where sessions are kept so a restart does not lose them (empty = memory only).')
    ap.add_argument('--preload', default='pisasr,codeformer',
                    help='models to warm up in the background after start (comma-separated keys, empty = none).')
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    import cv2
    import torch
    import uvicorn
    from .server import create_app

    cv2.setNumThreads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)

    if args.max_gpu_memory:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_memory * 1e9 / total), 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    app = create_app(args.weights, 'cuda', args.gpu_budget, args.idle_offload,
                     [k for k in args.preload.split(',') if k], args.session_dir or None, args.max_gpu_memory or None)
    host = socket.gethostname()
    print(f'\nRAP (Restore Anything Pipeline) is running on http://{args.host}:{args.port}')
    if args.host in ('127.0.0.1', 'localhost'):
        print(f'From your laptop:  ssh -N -L {args.port}:localhost:{args.port} {host}   then open http://localhost:{args.port}\n')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
