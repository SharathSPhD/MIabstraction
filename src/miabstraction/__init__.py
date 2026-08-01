"""MIabstraction: empirical validation of the transformer abstraction-layer hypothesis.

Sets the cuBLAS workspace config at import time — it must be in the environment before
CUDA initializes, or `torch.use_deterministic_algorithms` cannot make matmuls
reproducible. Without it, repeated runs of the same config drift (we observed a probe
accuracy move 0.786 -> 0.857 across identical runs).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
