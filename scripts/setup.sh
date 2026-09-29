#!/usr/bin/env bash
# --------------------------------------------
# Environment setup
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

# Create the conda env, fetch third-party code at pinned commits, and download weights.
#   bash scripts/setup.sh            # env name defaults to "rap"
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-rap}

clone() {  # clone <dir> <url> <commit>
  [ -d "third_party/$1" ] || git clone -q "$2" "third_party/$1"
  git -C "third_party/$1" checkout -q "$3"
}
mkdir -p third_party
clone sam3 https://github.com/facebookresearch/sam3.git 2345a4ad109ac29c569da749c91d84f10dc08c40
clone CodeFormer https://github.com/sczhou/CodeFormer.git b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b
clone ObjectClear https://github.com/zjx0101/ObjectClear.git c052d91ecd8772744a5ab97527441c813ab83009
clone DDColor https://github.com/piddnad/DDColor.git 2adb63f2656ac41cbdf7b894cddd94121a3faf13
clone HVI-CIDNet https://github.com/Fediory/HVI-CIDNet.git eb43d7d91e9a336c66856824ff9e4603ae41f408

if ! conda env list | grep -q "^$ENV "; then
  conda create -y -q -n "$ENV" python=3.12
fi
PY=$(conda run -n "$ENV" which python)
"$PY" -m pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r requirements.lock          # the exact versions everything is tested with
"$PY" -m pip install -q --no-deps -e third_party/sam3
"$PY" -m pip install -q --no-deps -e .

bash scripts/download_weights.sh "$PY"
echo
echo "Done. SAM 3 is gated: request access at https://huggingface.co/facebook/sam3 and run 'hf auth login' once."
echo "Start with:  conda activate $ENV && python -m rap --gpu 0"
