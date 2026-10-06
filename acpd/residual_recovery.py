import math

import numpy as np
from scipy.signal import butter, filtfilt

# --- Sensitivity-analysis hooks: defaults == frozen paper values; behavior is
# --- identical unless a caller overrides these module globals before invocation.
GATE_W_S = 0.78
GATE_W_F = 0.22
GATE_KAPPA_M = 1.25
GATE_B_M = 0.25


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2) + 1e-12))


def _bandpass(x, sr, low, high, order=4):
    nyq = 0.5 * sr
    high = min(high, nyq * 0.95)
    if low <= 0 or high <= low:
        return np.asarray(x, dtype=np.float32)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return np.asarray(x, dtype=np.float32)
    return filtfilt(b, a, x).astype(np.float32)


def _smooth_1d(x, win):
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.hanning(win)
    if np.sum(kernel) <= 0:
        return x
    kernel = kernel / np.sum(kernel)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _cycle_matrix(x, s1, sr, n_bins, min_cycle_sec, max_cycle_sec):
    s1 = np.asarray(s1, dtype=int)
    s1 = np.sort(s1[(s1 >= 0) & (s1 < len(x))])
    cycles = []
    rows = []
    phase_grid = np.linspace(0.0, 1.0, n_bins, endpoint=False)

    for start, end in zip(s1[:-1], s1[1:]):
        length = int(end - start)
        duration = length / sr
        if duration < min_cycle_sec or duration > max_cycle_sec:
            continue
        if length < 32:
            continue
        seg = np.asarray(x[start:end], dtype=float)
        src_phase = np.linspace(0.0, 1.0, len(seg), endpoint=False)
        rows.append(np.interp(phase_grid, src_phase, seg))
        cycles.append((int(start), int(end)))

    if not rows:
        return None, []
    return np.stack(rows, axis=0), cycles


def _robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def _phase_scores(matrix, score_smooth_bins=7):
    abs_x = np.abs(matrix)
    phase_energy = np.median(abs_x, axis=0)
    global_floor = np.median(abs_x) + 1e-12
    energy_ratio = phase_energy / global_floor

    # Sign-coherent residuals get a high waveform coherence. Turbulent murmurs
    # may have weaker sign coherence, so this is only one part of the score.
    coherence = np.abs(np.mean(matrix, axis=0)) / (np.mean(abs_x, axis=0) + 1e-12)

    hit_thr = np.median(abs_x, axis=0, keepdims=True) + 0.5 * (
        np.median(np.abs(abs_x - np.median(abs_x, axis=0, keepdims=True)), axis=0, keepdims=True) + 1e-12
    )
    hit_ratio = np.mean(abs_x >= hit_thr, axis=0)

    if matrix.shape[0] >= 3:
        adjacent_product = matrix[:-1] * matrix[1:]
        ck = np.mean(adjacent_product**2, axis=0) / (np.mean(matrix**2, axis=0) ** 2 + 1e-12)
    else:
        ck = np.ones(matrix.shape[1], dtype=float)

    score = np.log1p(np.maximum(energy_ratio - 1.0, 0.0)) * (0.35 + 0.65 * hit_ratio)
    score *= 0.5 + 0.5 * np.clip(coherence, 0.0, 1.0)
    score *= 0.5 + 0.5 * np.tanh(ck / 3.0)
    score = _smooth_1d(score, score_smooth_bins)
    return score, {
        "energy_ratio_mean": float(np.mean(energy_ratio)),
        "energy_ratio_p90": float(np.percentile(energy_ratio, 90)),
        "coherence_mean": float(np.mean(coherence)),
        "hit_ratio_mean": float(np.mean(hit_ratio)),
        "correlated_kurtosis_mean": float(np.mean(ck)),
    }


def _make_phase_mask(score, keep_quantile=0.82, min_z=0.5, smooth_bins=11):
    if len(score) == 0 or not np.isfinite(score).any():
        return np.zeros_like(score, dtype=float), 0.0
    z = _robust_z(score)
    q_thr = float(np.quantile(score, keep_quantile))
    raw = (score >= q_thr) & (z >= min_z)
    if not np.any(raw):
        return np.zeros_like(score, dtype=float), 0.0
    mask = raw.astype(float)
    mask = _smooth_1d(mask, smooth_bins)
    peak = float(np.max(mask))
    if peak > 1e-12:
        mask = mask / peak
    return mask.astype(np.float32), float(np.mean(raw))


def _phase_mask_to_signal(mask, cycles, n):
    out = np.zeros(n, dtype=np.float32)
    phase_grid = np.linspace(0.0, 1.0, len(mask), endpoint=False)
    for start, end in cycles:
        length = end - start
        if length <= 1:
            continue
        dst_phase = np.linspace(0.0, 1.0, length, endpoint=False)
        out[start:end] = np.interp(dst_phase, phase_grid, mask)
    return out


def _sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _subband_consistency(resid, sr, s1, n_bins, min_cycle_sec, max_cycle_sec):
    bands = [(80.0, 140.0), (140.0, 220.0), (220.0, 350.0)]
    band_rows = []
    cycles_ref = None
    for low, high in bands:
        xb = _bandpass(resid, sr, low, high)
        matrix, cycles = _cycle_matrix(np.abs(xb), s1, sr, n_bins, min_cycle_sec, max_cycle_sec)
        if matrix is None:
            continue
        if cycles_ref is None:
            cycles_ref = cycles
        if len(cycles) != len(cycles_ref):
            continue
        band_rows.append(matrix)

    if len(band_rows) < 2:
        return np.ones(n_bins, dtype=np.float32) * 0.5

    band_energy = np.stack(band_rows, axis=0) + 1e-12  # band, cycle, phase
    dominant_band = np.argmax(band_energy, axis=0)  # cycle, phase
    consistency = np.zeros(n_bins, dtype=float)
    for phase_idx in range(n_bins):
        counts = np.bincount(dominant_band[:, phase_idx], minlength=len(band_rows))
        consistency[phase_idx] = np.max(counts) / max(1, dominant_band.shape[0])

    mean_band_energy = np.mean(band_energy, axis=1)  # band, phase
    dominance = np.max(mean_band_energy, axis=0) / (np.sum(mean_band_energy, axis=0) + 1e-12)
    out = 0.55 * consistency + 0.45 * dominance
    return _smooth_1d(out, 9).astype(np.float32)


def _build_soft_gate(score, freq_consistency, keep_floor=0.06):
    z_score = _robust_z(score)
    z_freq = _robust_z(freq_consistency)
    evidence = GATE_W_S * z_score + GATE_W_F * z_freq
    evidence = _smooth_1d(evidence, 11)

    gate = _sigmoid(GATE_KAPPA_M * (evidence - GATE_B_M))
    # Soft-knee floor: tiny evidence remains nearly zero, but strong evidence
    # does not jump discontinuously as a hard mask would.
    gate = np.where(evidence > -0.25, gate, keep_floor * gate)
    gate = _smooth_1d(gate, 13)
    gate = np.clip(gate, 0.0, 1.0)
    return gate.astype(np.float32), evidence.astype(np.float32)


def recover_periodic_residual_v2(
    x_hat,
    resid,
    sr,
    s1,
    f_low=80.0,
    f_high=350.0,
    n_bins=256,
    alpha_max=0.85,
    min_cycles=4,
    min_cycle_sec=0.35,
    max_cycle_sec=1.6,
    max_added_rms_ratio=0.50,
):
    """Soft residual gate inspired by graph-cycle denoising, MCKD, and IRM.

    Compared with recover_periodic_residual(), this version avoids a binary
    phase mask. It fuses three paper-style cues:
    - cycle-level residual concentration,
    - correlated-kurtosis-like periodicity,
    - sub-band frequency consistency.

    The final residual gain is continuous and evidence-adaptive.
    """
    x_hat = np.asarray(x_hat, dtype=np.float32)
    resid = np.asarray(resid, dtype=np.float32)

    resid_hf = _bandpass(resid, sr, f_low, f_high)
    matrix, cycles = _cycle_matrix(resid_hf, s1, sr, n_bins, min_cycle_sec, max_cycle_sec)
    if matrix is None or len(cycles) < min_cycles:
        return x_hat.copy(), {
            "decision": "too_few_cycles",
            "num_cycles": len(cycles),
            "added_rms_ratio": 0.0,
            "alpha_eff": 0.0,
        }

    score, score_info = _phase_scores(matrix, score_smooth_bins=9)
    freq_consistency = _subband_consistency(resid_hf, sr, s1, n_bins, min_cycle_sec, max_cycle_sec)
    gate_phase, evidence = _build_soft_gate(score, freq_consistency)

    top_evidence = float(np.mean(np.sort(evidence)[-max(4, n_bins // 10) :]))
    concentration = float((np.percentile(score, 90) + 1e-12) / (np.median(score) + 1e-12))
    freq_strength = float(np.mean(np.sort(freq_consistency)[-max(4, n_bins // 10) :]))
    cycle_confidence = float(_sigmoid(0.6 * (len(cycles) - 10)))
    residual_strength = float(_rms(resid_hf) / (_rms(x_hat) + 1e-12))

    # Adaptive alpha mirrors speech-enhancement ratio masks: strong, localized
    # periodic evidence gets more residual; diffuse noise gets less.
    alpha_eff = alpha_max * _sigmoid(0.9 * top_evidence + 1.2 * (concentration - 1.4) + 1.0 * (freq_strength - 0.62))
    alpha_eff *= 0.4 + 0.6 * cycle_confidence
    alpha_eff *= float(_sigmoid(3.0 * (residual_strength - 0.18)))
    alpha_eff = float(np.clip(alpha_eff, 0.05, alpha_max))

    sample_gate = _phase_mask_to_signal(gate_phase, cycles, len(resid))
    periodic_layer = alpha_eff * sample_gate * resid_hf

    evidence_rms_cap = max_added_rms_ratio * _sigmoid(0.8 * top_evidence + 0.7 * (concentration - 1.2))
    evidence_rms_cap *= 0.45 + 0.55 * cycle_confidence
    evidence_rms_cap *= float(_sigmoid(2.2 * (residual_strength - 0.16)))
    evidence_rms_cap = float(np.clip(evidence_rms_cap, 0.12, max_added_rms_ratio))
    max_rms = evidence_rms_cap * (_rms(x_hat) + 1e-12)
    layer_rms = _rms(periodic_layer)
    if layer_rms > max_rms:
        periodic_layer = periodic_layer * (max_rms / (layer_rms + 1e-12))
        layer_rms = _rms(periodic_layer)

    y = (x_hat + periodic_layer).astype(np.float32)
    return y, {
        "decision": "soft_periodic_gate_added" if layer_rms > 1e-8 else "soft_periodic_gate_zero",
        "num_cycles": len(cycles),
        "gate_mean": float(np.mean(gate_phase)),
        "gate_p90": float(np.percentile(gate_phase, 90)),
        "added_rms_ratio": float(layer_rms / (_rms(x_hat) + 1e-12)),
        "alpha_eff": alpha_eff,
        "evidence_rms_cap": evidence_rms_cap,
        "top_evidence": top_evidence,
        "concentration": concentration,
        "freq_strength": freq_strength,
        "cycle_confidence": cycle_confidence,
        "residual_strength": residual_strength,
        "freq_consistency_mean": float(np.mean(freq_consistency)),
        **score_info,
    }


def recover_cycle_consistent_residual(
    x_hat,
    resid,
    sr,
    s1,
    f_low=80.0,
    f_high=350.0,
    n_bins=256,
    alpha_max=0.85,
    min_cycles=4,
    min_cycle_sec=0.35,
    max_cycle_sec=1.6,
    max_added_rms_ratio=0.50,
):
    """Recover diagnostically relevant residuals consistent across S1--S1 cycles.

    This paper-facing wrapper maps the template residual to a normalized
    cardiac-cycle phase grid, estimates a continuous phase soft gate from
    cross-cycle support and sub-band consistency, and adds back a bounded
    residual layer. It preserves the previous implementation by delegating to
    ``recover_periodic_residual_v2``.
    """
    return recover_periodic_residual_v2(
        x_hat=x_hat,
        resid=resid,
        sr=sr,
        s1=s1,
        f_low=f_low,
        f_high=f_high,
        n_bins=n_bins,
        alpha_max=alpha_max,
        min_cycles=min_cycles,
        min_cycle_sec=min_cycle_sec,
        max_cycle_sec=max_cycle_sec,
        max_added_rms_ratio=max_added_rms_ratio,
    )


def recover_periodic_residual(
    x_hat,
    resid,
    sr,
    s1,
    f_low=80.0,
    f_high=350.0,
    n_bins=256,
    alpha=0.45,
    keep_quantile=0.82,
    min_z=0.5,
    min_cycles=4,
    min_cycle_sec=0.35,
    max_cycle_sec=1.6,
    max_added_rms_ratio=0.35,
):
    """Recover residual components that repeat at a stable cardiac-cycle phase.

    This is an experimental TSA/MCKD-inspired gate: it estimates a phase mask
    from the high-frequency residual matrix aligned by S1-S1 cycles, then adds
    only masked residual samples back to the template reconstruction.
    """
    x_hat = np.asarray(x_hat, dtype=np.float32)
    resid = np.asarray(resid, dtype=np.float32)

    resid_hf = _bandpass(resid, sr, f_low, f_high)
    matrix, cycles = _cycle_matrix(resid_hf, s1, sr, n_bins, min_cycle_sec, max_cycle_sec)
    if matrix is None or len(cycles) < min_cycles:
        return x_hat.copy(), {
            "decision": "too_few_cycles",
            "num_cycles": len(cycles),
            "added_rms_ratio": 0.0,
        }

    score, score_info = _phase_scores(matrix)
    phase_mask, mask_ratio = _make_phase_mask(score, keep_quantile=keep_quantile, min_z=min_z)
    if mask_ratio <= 0:
        return x_hat.copy(), {
            "decision": "no_periodic_phase",
            "num_cycles": len(cycles),
            "mask_ratio": 0.0,
            "added_rms_ratio": 0.0,
            **score_info,
        }

    sample_mask = _phase_mask_to_signal(phase_mask, cycles, len(resid))
    periodic_layer = alpha * sample_mask * resid_hf

    # Keep the added layer bounded relative to the template. This prevents the
    # gate from re-injecting too much mixed noise when periodic evidence is weak.
    max_rms = max_added_rms_ratio * (_rms(x_hat) + 1e-12)
    layer_rms = _rms(periodic_layer)
    if layer_rms > max_rms:
        periodic_layer = periodic_layer * (max_rms / (layer_rms + 1e-12))
        layer_rms = _rms(periodic_layer)

    y = (x_hat + periodic_layer).astype(np.float32)
    decision = "periodic_residual_added" if layer_rms > 1e-8 else "periodic_layer_zero"
    return y, {
        "decision": decision,
        "num_cycles": len(cycles),
        "mask_ratio": float(mask_ratio),
        "added_rms_ratio": float(layer_rms / (_rms(x_hat) + 1e-12)),
        "alpha": float(alpha),
        "keep_quantile": float(keep_quantile),
        "min_z": float(min_z),
        **score_info,
    }
