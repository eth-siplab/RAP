# --------------------------------------------
# RAP package setup
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

import os

# numpy madvise()s large arrays for transparent huge pages. On machines with
# fragmented memory and THP defrag=madvise, every such allocation can stall
# for tens of seconds in kernel memory compaction (seen on shared servers).
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
try:  # in case numpy was imported before us
    import numpy.core.multiarray as _ma
    _ma._set_madvise_hugepage(False)
except Exception:
    pass
