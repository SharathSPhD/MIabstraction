"""Determinism control.

SPEC.md requires that a config plus a seed reproduce a result. GPU training defaults to
non-deterministic kernels, so identical runs drift; `set_determinism` pins every source
of randomness we control and asks torch for deterministic algorithms.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_determinism(seed: int, strict: bool = True) -> None:
    """Seed all RNGs and request deterministic kernels.

    strict=False falls back to seeding only, for ops with no deterministic
    implementation (raises otherwise).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        # The memory-efficient and flash attention kernels have non-deterministic
        # backward passes; the math backend does not. Attention is the whole model
        # here, so this is the difference between reproducible and not.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
