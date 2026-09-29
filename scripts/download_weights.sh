#!/usr/bin/env bash
# --------------------------------------------
# Model weight downloads
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

# Download model weights into weights/. SAM 3 and SD-2.1-base are fetched from
# Hugging Face on first start instead.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python}
mkdir -p weights/pisasr weights/codeformer weights/fbcnn

# PiSA-SR LoRA weights (CVPR 2025, Apache-2.0)
[ -f weights/pisasr/pisa_sr.pkl ] || "$PY" -m gdown --folder \
  https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss -O weights/pisasr

# CodeFormer + its face detector and parser (S-Lab License 1.0, non-commercial)
for f in codeformer.pth detection_Resnet50_Final.pth parsing_parsenet.pth; do
  [ -f "weights/codeformer/$f" ] || wget -q --show-progress -O "weights/codeformer/$f" \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/$f"
done
mkdir -p third_party/CodeFormer/weights/facelib
for f in detection_Resnet50_Final.pth parsing_parsenet.pth; do
  ln -sfn "../../../../weights/codeformer/$f" "third_party/CodeFormer/weights/facelib/$f"
done

# LaMa (TorchScript export from IOPaint, Apache-2.0)
mkdir -p weights/lama weights/nafnet
[ -f weights/lama/big-lama.pt ] || wget -q --show-progress -O weights/lama/big-lama.pt \
  https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt

# NAFNet GoPro (ECCV 2022, MIT), official Google Drive file
[ -f weights/nafnet/NAFNet-GoPro-width64.pth ] || "$PY" -m gdown 1S0PVRbyTakYY9a82kujgZLbMihfNBLfC \
  -O weights/nafnet/NAFNet-GoPro-width64.pth

# SCUNet real-noise models (Apache-2.0) and DRUNet (MIT), from the KAIR releases
mkdir -p weights/kair
for f in scunet_color_real_psnr.pth scunet_color_real_gan.pth drunet_color.pth; do
  [ -f weights/kair/$f ] || wget -q --show-progress -O weights/kair/$f https://github.com/cszn/KAIR/releases/download/v1.0/$f
done

# ObjectClear (S-Lab License, non-commercial), DDColor (Apache-2.0) and
# HVI-CIDNet (MIT) come from Hugging Face on first use.

# FBCNN (ICCV 2021, Apache-2.0), JPEG artifact removal
[ -f weights/fbcnn/fbcnn_deblock.pth ] || wget -q --show-progress -O weights/fbcnn/fbcnn_deblock.pth \
  https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth
