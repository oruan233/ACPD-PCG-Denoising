"""ACPD: Adaptive Cycle-Prior Denoising.

A training-free heart-sound denoiser using coarse cardiac-cycle anchors.
The default is the data-adaptive JBHI main-table method; version="fixed"
selects the historical frozen core. See docs/VERSIONS.md for source provenance.

Public API
----------
denoise_array(noisy, sr) -> (denoised, info)
    Denoise an audio array, preserving input sample rate and frame count.
denoise_file(input_path, output_path) -> dict
    Denoise a single WAV file.
run_baseline(clean, noisy, sr, ...) -> dict
    Low-level entry point; the ACPD output is ``result["adaptive_v3"]``.
compute_metrics(clean, noisy, denoised, sr) -> dict
    Reference-based metrics (delta SNR, delta SI-SDR, RMSE, MAE, ...).

Notes
-----
The frozen core sources (``adaptive_v4.py``, ``residual_recovery.py``,
``legacy_pcg_core.py``, ``Detect_s1_s2_new.py``) use bare
intra-directory imports. This ``__init__`` puts the package directory on
``sys.path`` so those imports resolve without editing the algorithm sources.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from .denoise_wav import denoise_array, denoise_file, DEFAULT_METHOD
from .pipeline import run_baseline, compute_metrics, VERSIONS

__all__ = [
    "denoise_array",
    "denoise_file",
    "run_baseline",
    "compute_metrics",
    "DEFAULT_METHOD",
    "VERSIONS",
]
__version__ = "0.2.0"
