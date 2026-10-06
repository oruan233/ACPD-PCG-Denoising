#!/usr/bin/env python3
"""Smoke tests for ACPD that need no external data.

Runs the method on a synthetic PCG-like signal and checks that it produces a
finite, same-length waveform. Run with pytest, or directly:

    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import acpd  # noqa: E402


def make_synthetic_pcg(sr: int = 2000, seconds: float = 8.0, hr_bpm: float = 75.0) -> np.ndarray:
    """A crude but periodic S1/S2 heart-sound surrogate (no real data)."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    x = np.zeros(n, dtype=float)
    period = 60.0 / hr_bpm
    n_cycles = int(seconds / period)
    for k in range(n_cycles):
        t1 = k * period
        t2 = t1 + 0.32
        x += np.exp(-((t - t1) / 0.020) ** 2) * np.sin(2 * np.pi * 50 * (t - t1))
        x += 0.7 * np.exp(-((t - t2) / 0.018) ** 2) * np.sin(2 * np.pi * 70 * (t - t2))
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


def test_denoise_array_runs() -> None:
    sr = 2000
    clean = make_synthetic_pcg(sr)
    rng = np.random.default_rng(0)
    noisy = clean + 0.3 * rng.standard_normal(clean.shape)

    denoised, info = acpd.denoise_array(noisy, sr)

    assert denoised.shape == noisy.shape, "output length must match input"
    assert np.all(np.isfinite(denoised)), "output must be finite"
    assert isinstance(info, dict), "second return value should be run metadata"


def test_default_method_is_acpd() -> None:
    assert acpd.DEFAULT_METHOD in {"adaptive_v3", "acpd"}


if __name__ == "__main__":
    test_denoise_array_runs()
    test_default_method_is_acpd()
    print("smoke tests passed")
