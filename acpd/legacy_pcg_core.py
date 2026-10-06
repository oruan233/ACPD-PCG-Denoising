"""Legacy PCG primitives required by AdaptiveV4.

This file keeps the mathematically active parts of the original project that
AdaptiveV3/V4 depend on:
- S1/S2 template reconstruction.
- Phase-dominant high-frequency residual recovery.

Unused plotting, dataset, and exploratory functions were intentionally removed
from the standalone deployment package.
"""

from __future__ import annotations

import numpy as np


def _extract_snippets(x, idx_list, half_len):
    snippets = []
    valid_idx = []
    for p in idx_list:
        if p - half_len < 0 or p + half_len >= len(x):
            continue
        snippets.append(x[p - half_len:p + half_len + 1])
        valid_idx.append(p)
    if len(snippets) == 0:
        return None, None
    return np.stack(snippets, axis=0), np.array(valid_idx, dtype=int)


def _robust_template(snips, mode="median", trim=0.2):
    if mode == "median":
        return np.median(snips, axis=0)
    sn = np.sort(snips, axis=0)
    n = sn.shape[0]
    lo = int(np.floor(trim * n))
    hi = int(np.ceil((1 - trim) * n))
    return np.mean(sn[lo:hi], axis=0)


def _refine_by_template(x, idx_list, template, search_radius):
    length = len(template)
    half = length // 2
    temp = template - np.mean(template)

    refined = []
    for p in idx_list:
        lo = max(half, p - search_radius)
        hi = min(len(x) - half - 1, p + search_radius)
        best_p = p
        best_score = -1e18
        for q in range(lo, hi + 1):
            seg = x[q - half:q + half + 1]
            seg = seg - np.mean(seg)
            score = float(np.dot(seg, temp) / (np.linalg.norm(seg) * np.linalg.norm(temp) + 1e-12))
            if score > best_score:
                best_score = score
                best_p = q
        refined.append(best_p)
    return np.array(refined, dtype=int)


def _estimate_amplitude(x, p, template):
    length = len(template)
    half = length // 2
    seg = x[p - half:p + half + 1]
    return float(np.dot(seg, template) / (np.dot(template, template) + 1e-12))


def denoise_pcg_by_s1s2_templates(
    x,
    sr,
    s1_idx,
    s2_idx,
    win_ms=80,
    refine_ms=80,
    template_mode="median",
    trim=0.2,
    iters=2,
):
    half_len = int((win_ms / 1000.0) * sr)
    search_radius = int((refine_ms / 1000.0) * sr)

    s1_ref = np.array(s1_idx, dtype=int)
    s2_ref = np.array(s2_idx, dtype=int)

    tpl_s1 = None
    tpl_s2 = None
    for _ in range(iters):
        sn1, s1_valid = _extract_snippets(x, s1_ref, half_len)
        sn2, s2_valid = _extract_snippets(x, s2_ref, half_len)
        if sn1 is None or sn2 is None:
            return None, None, None, None, None, None

        tpl_s1 = _robust_template(sn1, mode=template_mode, trim=trim)
        tpl_s2 = _robust_template(sn2, mode=template_mode, trim=trim)
        s1_ref = _refine_by_template(x, s1_valid, tpl_s1, search_radius)
        s2_ref = _refine_by_template(x, s2_valid, tpl_s2, search_radius)

    x_hat = np.zeros_like(x, dtype=float)
    length = len(tpl_s1)
    half = length // 2

    for p in s1_ref:
        if p - half < 0 or p + half >= len(x):
            continue
        x_hat[p - half:p + half + 1] += _estimate_amplitude(x, p, tpl_s1) * tpl_s1

    for p in s2_ref:
        if p - half < 0 or p + half >= len(x):
            continue
        x_hat[p - half:p + half + 1] += _estimate_amplitude(x, p, tpl_s2) * tpl_s2

    resid = x - x_hat
    return x_hat.astype(np.float32), resid.astype(np.float32), s1_ref, s2_ref, tpl_s1, tpl_s2


def keep_band_freq_only(x, sr, f_low=100, f_high=200):
    x = x - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    mask = (freqs >= f_low) & (freqs <= f_high)
    return np.fft.irfft(spectrum * mask, n=n)


def dominant_freq_in_band(x, sr, f_low=100, f_high=300):
    if len(x) < 16:
        return None
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return None
    f_band = freqs[mask]
    x_band = spectrum[mask]
    return f_band[np.argmax(x_band)]


def recover_hf_by_phase_dominance_with_freq_consistency(
    resid,
    sr,
    s1,
    s2,
    f_low=100,
    f_high=300,
    dominance_ratio=0.6,
    exclude_ms=40,
    freq_tol=20,
):
    resid_hf = keep_band_freq_only(resid, sr, f_low, f_high)
    num_cycles = min(len(s1), len(s2)) - 1
    if num_cycles < 3:
        return np.zeros_like(resid), {"decision": "too_few_cycles"}

    margin = int(exclude_ms * sr / 1000)
    sys_wins, dia_wins = 0, 0
    sys_freqs, dia_freqs = [], []
    valid_cycles = 0

    for k in range(num_cycles):
        s1k, s2k, s1n = s1[k], s2[k], s1[k + 1]
        a = min(s1k, s2k) + margin
        b = max(s1k, s2k) - margin
        c = min(s2k, s1n) + margin
        d = max(s2k, s1n) - margin
        if b <= a or d <= c:
            continue

        sys_seg = resid_hf[a:b]
        dia_seg = resid_hf[c:d]
        e_sys = np.mean(sys_seg**2) if len(sys_seg) else 0
        e_dia = np.mean(dia_seg**2) if len(dia_seg) else 0
        if e_sys == 0 and e_dia == 0:
            continue

        valid_cycles += 1
        if e_sys > e_dia:
            sys_wins += 1
            freq = dominant_freq_in_band(sys_seg, sr, f_low, f_high)
            if freq is not None:
                sys_freqs.append(freq)
        else:
            dia_wins += 1
            freq = dominant_freq_in_band(dia_seg, sr, f_low, f_high)
            if freq is not None:
                dia_freqs.append(freq)

    if valid_cycles == 0:
        return np.zeros_like(resid), {"decision": "no_valid_cycles"}

    sys_ratio = sys_wins / valid_cycles
    dia_ratio = dia_wins / valid_cycles
    x_hf_all = np.zeros_like(resid)

    if sys_ratio >= dominance_ratio and len(sys_freqs) >= 3:
        f_std = np.std(sys_freqs)
        if f_std <= freq_tol:
            for k in range(num_cycles):
                a = min(s1[k], s2[k]) + margin
                b = max(s1[k], s2[k]) - margin
                if b > a:
                    x_hf_all[a:b] += resid_hf[a:b]
            decision = "systolic_murmur"
        else:
            decision = "noise_like"
    elif dia_ratio >= dominance_ratio and len(dia_freqs) >= 3:
        f_std = np.std(dia_freqs)
        if f_std <= freq_tol:
            for k in range(num_cycles):
                c = min(s2[k], s1[k + 1]) + margin
                d = max(s2[k], s1[k + 1]) - margin
                if d > c:
                    x_hf_all[c:d] += resid_hf[c:d]
            decision = "diastolic_murmur"
        else:
            decision = "noise_like"
    else:
        decision = "noise_like"

    return x_hf_all, {
        "num_cycles": valid_cycles,
        "sys_ratio": sys_ratio,
        "dia_ratio": dia_ratio,
        "mean_sys_freq": np.mean(sys_freqs) if sys_freqs else None,
        "mean_dia_freq": np.mean(dia_freqs) if dia_freqs else None,
        "sys_freq_std": np.std(sys_freqs) if sys_freqs else None,
        "dia_freq_std": np.std(dia_freqs) if dia_freqs else None,
        "freq_tol": freq_tol,
        "decision": decision,
    }
