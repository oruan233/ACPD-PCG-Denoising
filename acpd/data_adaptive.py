"""Research data-adaptive parameterizations, including the JBHI main-table method.

Mathematics ported from experiments/acpd_data_adaptive.py. This module temporarily
replaces the cycle-consistent residual callback and restores it before returning.
Use the synchronized public acpd.run_baseline entry point for concurrent callers.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np


from . import adaptive_v4 as frozen_acpd
from . import residual_recovery as frozen_rr


VARIANTS = ("fixed", "adaptive_gate", "adaptive_residual", "adaptive_full")


def _upper_tail_reliability(z: np.ndarray) -> float:
    """Robust positive-tail contrast used as within-recording reliability."""
    z = np.asarray(z, dtype=float)
    value = np.quantile(z, 0.90) - np.median(z)
    return float(max(value, 1e-6))


def _adaptive_soft_gate(score: np.ndarray, freq_consistency: np.ndarray):
    z_score = frozen_rr._robust_z(score)
    z_freq = frozen_rr._robust_z(freq_consistency)

    rel_score = _upper_tail_reliability(z_score)
    rel_freq = _upper_tail_reliability(z_freq)
    weight_score = rel_score / (rel_score + rel_freq)
    weight_freq = 1.0 - weight_score
    evidence = weight_score * z_score + weight_freq * z_freq
    evidence = frozen_rr._smooth_1d(evidence, 11)

    q10, q90 = np.quantile(evidence, [0.10, 0.90])
    center = float(0.5 * (q10 + q90))
    width = float(max(q90 - q10, 1e-6))
    slope = float(2.0 * math.log(9.0) / width)
    gate = frozen_rr._sigmoid(slope * (evidence - center))
    gate = frozen_rr._smooth_1d(gate, 13)
    gate = np.clip(gate, 0.0, 1.0)
    return gate.astype(np.float32), evidence.astype(np.float32), {
        "adaptive_weight_score": weight_score,
        "adaptive_weight_freq": weight_freq,
        "adaptive_gate_center": center,
        "adaptive_gate_slope": slope,
    }


def _fixed_alpha_and_cap(
    x_hat,
    resid_hf,
    sample_gate,
    top_evidence,
    concentration,
    freq_strength,
    cycle_confidence,
    residual_strength,
    alpha_max,
    max_added_rms_ratio,
):
    alpha_eff = alpha_max * frozen_rr._sigmoid(
        0.9 * top_evidence
        + 1.2 * (concentration - 1.4)
        + (freq_strength - 0.62)
    )
    alpha_eff *= 0.4 + 0.6 * cycle_confidence
    alpha_eff *= float(frozen_rr._sigmoid(3.0 * (residual_strength - 0.18)))
    alpha_eff = float(np.clip(alpha_eff, 0.05, alpha_max))

    layer = alpha_eff * sample_gate * resid_hf
    cap = max_added_rms_ratio * frozen_rr._sigmoid(
        0.8 * top_evidence + 0.7 * (concentration - 1.2)
    )
    cap *= 0.45 + 0.55 * cycle_confidence
    cap *= float(frozen_rr._sigmoid(2.2 * (residual_strength - 0.16)))
    cap = float(np.clip(cap, 0.12, max_added_rms_ratio))
    return layer, alpha_eff, cap


def _projected_alpha_and_cap(
    x_hat,
    resid_hf,
    matrix,
    cycles,
    sample_gate,
    alpha_max,
    max_added_rms_ratio,
):
    candidate = sample_gate * resid_hf
    consensus_phase = np.median(matrix, axis=0)
    consensus = frozen_rr._phase_mask_to_signal(consensus_phase, cycles, len(resid_hf))
    valid = sample_gate > 1e-3
    if np.any(valid):
        numerator = float(np.dot(candidate[valid], consensus[valid]))
        denominator = float(np.dot(candidate[valid], candidate[valid]) + 1e-12)
        alpha_eff = float(np.clip(numerator / denominator, 0.0, alpha_max))
    else:
        alpha_eff = 0.0

    layer = alpha_eff * candidate
    consensus_ratio = frozen_rr._rms(consensus) / (frozen_rr._rms(x_hat) + 1e-12)
    cap = float(np.clip(consensus_ratio, 0.0, max_added_rms_ratio))
    return layer, alpha_eff, cap


def recover_cycle_consistent_residual_adaptive(
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
    mode="adaptive_residual",
):
    if mode not in {"adaptive_gate", "adaptive_residual", "adaptive_full"}:
        raise ValueError(f"unsupported adaptive mode: {mode}")

    x_hat = np.asarray(x_hat, dtype=np.float32)
    resid = np.asarray(resid, dtype=np.float32)
    resid_hf = frozen_rr._bandpass(resid, sr, f_low, f_high)
    matrix, cycles = frozen_rr._cycle_matrix(
        resid_hf, s1, sr, n_bins, min_cycle_sec, max_cycle_sec
    )
    if matrix is None or len(cycles) < min_cycles:
        return x_hat.copy(), {
            "decision": "too_few_cycles",
            "num_cycles": len(cycles),
            "added_rms_ratio": 0.0,
            "alpha_eff": 0.0,
            "parameter_mode": mode,
        }

    score, score_info = frozen_rr._phase_scores(matrix, score_smooth_bins=9)
    freq_consistency = frozen_rr._subband_consistency(
        resid_hf, sr, s1, n_bins, min_cycle_sec, max_cycle_sec
    )
    gate_phase, evidence, gate_info = _adaptive_soft_gate(score, freq_consistency)

    top_evidence = float(np.mean(np.sort(evidence)[-max(4, n_bins // 10) :]))
    concentration = float(
        (np.percentile(score, 90) + 1e-12) / (np.median(score) + 1e-12)
    )
    freq_strength = float(
        np.mean(np.sort(freq_consistency)[-max(4, n_bins // 10) :])
    )
    cycle_confidence = float(frozen_rr._sigmoid(0.6 * (len(cycles) - 10)))
    residual_strength = float(
        frozen_rr._rms(resid_hf) / (frozen_rr._rms(x_hat) + 1e-12)
    )
    sample_gate = frozen_rr._phase_mask_to_signal(gate_phase, cycles, len(resid))

    if mode == "adaptive_gate":
        layer, alpha_eff, cap = _fixed_alpha_and_cap(
            x_hat,
            resid_hf,
            sample_gate,
            top_evidence,
            concentration,
            freq_strength,
            cycle_confidence,
            residual_strength,
            alpha_max,
            max_added_rms_ratio,
        )
    else:
        layer, alpha_eff, cap = _projected_alpha_and_cap(
            x_hat,
            resid_hf,
            matrix,
            cycles,
            sample_gate,
            alpha_max,
            max_added_rms_ratio,
        )

    max_rms = cap * (frozen_rr._rms(x_hat) + 1e-12)
    layer_rms = frozen_rr._rms(layer)
    if layer_rms > max_rms:
        layer = layer * (max_rms / (layer_rms + 1e-12))
        layer_rms = frozen_rr._rms(layer)

    y = (x_hat + layer).astype(np.float32)
    return y, {
        "decision": "adaptive_periodic_gate_added" if layer_rms > 1e-8 else "adaptive_periodic_gate_zero",
        "parameter_mode": mode,
        "num_cycles": len(cycles),
        "gate_mean": float(np.mean(gate_phase)),
        "gate_p90": float(np.percentile(gate_phase, 90)),
        "added_rms_ratio": float(layer_rms / (frozen_rr._rms(x_hat) + 1e-12)),
        "alpha_eff": alpha_eff,
        "evidence_rms_cap": cap,
        "top_evidence": top_evidence,
        "concentration": concentration,
        "freq_strength": freq_strength,
        "cycle_confidence": cycle_confidence,
        "residual_strength": residual_strength,
        "freq_consistency_mean": float(np.mean(freq_consistency)),
        **gate_info,
        **score_info,
    }


@contextmanager
def _temporary_residual_variant(mode: str):
    original = frozen_acpd.recover_cycle_consistent_residual

    def replacement(*args, **kwargs):
        return recover_cycle_consistent_residual_adaptive(*args, **kwargs, mode=mode)

    frozen_acpd.recover_cycle_consistent_residual = replacement
    try:
        yield
    finally:
        frozen_acpd.recover_cycle_consistent_residual = original


def _wiener_candidate_mix(noisy_band: np.ndarray, candidate: np.ndarray) -> float:
    discrepancy_power = frozen_acpd.rms(noisy_band - candidate) ** 2
    candidate_power = frozen_acpd.rms(candidate) ** 2
    return float(
        np.clip(
            discrepancy_power / (discrepancy_power + candidate_power + 1e-12),
            0.0,
            1.0,
        )
    )


def run_variant(
    clean,
    noisy,
    sr,
    mode="fixed",
    algorithm_input_bandpass=True,
    segmenter="existing",
    gate_floor=0.3,
):
    if mode not in VARIANTS:
        raise ValueError(f"unknown ACPD parameter mode: {mode}")
    if mode == "fixed":
        result = frozen_acpd.run_baseline(
            clean,
            noisy,
            sr,
            algorithm_input_bandpass=algorithm_input_bandpass,
            segmenter=segmenter,
            gate_floor=gate_floor,
        )
        result["parameter_mode"] = mode
        return result

    with _temporary_residual_variant(mode):
        result = frozen_acpd.run_baseline(
            clean,
            noisy,
            sr,
            algorithm_input_bandpass=algorithm_input_bandpass,
            segmenter=segmenter,
            gate_floor=gate_floor,
        )

    if mode == "adaptive_full" and "periodic_v2" in result:
        noisy_band = (
            frozen_acpd.bandpass_filter(noisy, sr)
            if algorithm_input_bandpass
            else np.asarray(noisy, dtype=np.float32)
        )
        candidate = np.asarray(result["periodic_v2"], dtype=np.float32)
        weight = _wiener_candidate_mix(noisy_band, candidate)
        mixed = weight * candidate + (1.0 - weight) * noisy_band
        s1 = np.rint(np.asarray(result["s1_times_sec"]) * sr).astype(int)
        s2 = np.rint(np.asarray(result["s2_times_sec"]) * sr).astype(int)
        result["adaptive_v3"] = frozen_acpd._phase_aware_gate(
            mixed, s1, s2, sr, floor=gate_floor
        )
        result["adaptive_v3_weight"] = weight
        result["adaptive_v3_outband_gate"] = math.nan
        result["adaptive_v3_residual_gate"] = math.nan

    result["parameter_mode"] = mode
    return result

