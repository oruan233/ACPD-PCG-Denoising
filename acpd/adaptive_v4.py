"""AdaptiveV4 denoising pipeline.

Algorithm flow:
1. Run AdaptiveV3 core reconstruction to obtain template, periodic, residual,
   and cycle-confidence signals.
2. Build conservative residual candidates around pathological-suspect phase windows.
3. Fuse the AdaptiveV3 base and residual candidates with a diagnostic-risk gate.

The deployable CLI uses `adaptive_v4_diagnostic_risk_fusion` by default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Detect_s1_s2_new import detect_s1_s2  # noqa: E402
from legacy_pcg_core import (  # noqa: E402
    denoise_pcg_by_s1s2_templates,
    recover_hf_by_phase_dominance_with_freq_consistency,
)
from residual_recovery import (  # noqa: E402
    recover_cycle_consistent_residual,
    recover_periodic_residual,
    recover_periodic_residual_v2,
)



# ============================================================
# AdaptiveV3 core reconstruction
# ============================================================
TARGET_SR = 2000
LOWCUT = 25.0
HIGHCUT = 400.0


@dataclass(frozen=True)
class AdaptiveV4Config:
    """Central configuration for the paper-aligned AdaptiveV4 E/F/G modules."""

    cycle_residual_band: tuple[float, float] = (80.0, 350.0)
    heart_band: tuple[float, float] = (20.0, 150.0)
    murmur_low_band: tuple[float, float] = (150.0, 250.0)
    murmur_mid_band: tuple[float, float] = (250.0, 400.0)
    murmur_tail_band: tuple[float, float] = (400.0, 600.0)
    gate_max: float = 0.85
    diagnostic_risk_lambda: float = 0.22
    anchor_base_weight: float = 0.60
    anchor_guard_weight: float = 0.40
    diagnostic_guard_alpha: float = 0.16
    diagnostic_dominance_ratio: float = 1.15


DEFAULT_ADAPTIVE_V4_CONFIG = AdaptiveV4Config()


def fs_path(path):
    path = str(path)
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith("\\\\?\\"):
            return "\\\\?\\" + path
    return path


def path_exists(path):
    return os.path.exists(fs_path(path))


def safe_walk(root):
    def on_error(error):
        return None

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        yield Path(dirpath), filenames


def read_csv_rows(path, has_header=True):
    with open(fs_path(path), "r", encoding="utf-8-sig", newline="") as f:
        if has_header:
            yield from csv.DictReader(f)
        else:
            reader = csv.reader(f)
            for row in reader:
                yield row


def find_files(root, name):
    for dirpath, filenames in safe_walk(root):
        for filename in filenames:
            if filename == name:
                yield dirpath / filename


def discover_archive(root):
    archive_dir = root / "archive"
    records = []
    if not path_exists(archive_dir):
        return records

    for csv_name in ("set_a.csv", "set_b.csv"):
        csv_path = archive_dir / csv_name
        if not path_exists(csv_path):
            continue
        for row in read_csv_rows(csv_path):
            label = (row.get("label") or "").strip()
            if not label:
                continue
            rel_audio = (row.get("fname") or "").strip().replace("/", os.sep)
            audio_path = archive_dir / rel_audio
            if not path_exists(audio_path):
                continue
            records.append(
                {
                    "dataset": "pascal_archive",
                    "record_id": Path(rel_audio).stem,
                    "audio_path": str(audio_path),
                    "source_subset": csv_name.replace(".csv", ""),
                    "label_raw": label,
                    "label_group": label,
                }
            )
    return records


def discover_physionet(root):
    records = []
    for ref_path in find_files(root, "REFERENCE.csv"):
        parts = {p.lower() for p in ref_path.parts}
        if not any(p.startswith("training-") or p == "validation" for p in parts):
            continue
        subset = ref_path.parent.name
        for row in read_csv_rows(ref_path, has_header=False):
            if len(row) < 2:
                continue
            rec_id = row[0].strip()
            label_raw = row[1].strip()
            if not rec_id or not label_raw:
                continue
            audio_path = ref_path.parent / f"{rec_id}.wav"
            if not path_exists(audio_path):
                continue
            label_group = "normal" if label_raw == "-1" else "abnormal"
            records.append(
                {
                    "dataset": "physionet2016",
                    "record_id": rec_id,
                    "audio_path": str(audio_path),
                    "source_subset": subset,
                    "label_raw": label_raw,
                    "label_group": label_group,
                }
            )
    return records


def discover_circor(root):
    records = []
    csv_paths = list(find_files(root, "training_data.csv"))
    if not csv_paths:
        return records

    data_csv = csv_paths[0]
    base_dir = data_csv.parent / "training_data"
    if not path_exists(base_dir):
        return records

    for row in read_csv_rows(data_csv):
        patient_id = str(row.get("Patient ID", "")).strip()
        if not patient_id:
            continue
        murmur = str(row.get("Murmur", "")).strip()
        outcome = str(row.get("Outcome", "")).strip()
        if murmur == "Unknown":
            continue
        pattern = f"{patient_id}_"
        audio_paths = []
        for dirpath, filenames in safe_walk(base_dir):
            for filename in filenames:
                if filename.startswith(pattern) and filename.lower().endswith(".wav"):
                    audio_paths.append(dirpath / filename)
        for audio_path in sorted(audio_paths):
            location = audio_path.stem.split("_")[-1]
            records.append(
                {
                    "dataset": "circor",
                    "record_id": audio_path.stem,
                    "audio_path": str(audio_path),
                    "source_subset": "training_data",
                    "label_raw": f"Murmur={murmur};Outcome={outcome}",
                    "label_group": f"murmur_{murmur.lower()}",
                    "patient_id": patient_id,
                    "location": location,
                }
            )
    return records


def discover_records(dataset_root):
    root = Path(dataset_root)
    records = []
    records.extend(discover_archive(root))
    records.extend(discover_physionet(root))
    records.extend(discover_circor(root))
    return records


def select_balanced(records, per_dataset, seed):
    rng = random.Random(seed)
    by_dataset = defaultdict(list)
    for rec in records:
        by_dataset[rec["dataset"]].append(rec)

    selected = []
    for dataset in sorted(by_dataset):
        by_label = defaultdict(list)
        for rec in by_dataset[dataset]:
            by_label[rec["label_group"]].append(rec)
        for items in by_label.values():
            rng.shuffle(items)

        labels = sorted(by_label, key=lambda k: len(by_label[k]), reverse=True)
        label_indices = {label: 0 for label in labels}
        picked = []
        while len(picked) < per_dataset:
            progressed = False
            for label in labels:
                idx = label_indices[label]
                if idx >= len(by_label[label]):
                    continue
                picked.append(by_label[label][idx])
                label_indices[label] += 1
                progressed = True
                if len(picked) >= per_dataset:
                    break
            if not progressed:
                break
        selected.extend(picked)
    return selected


def filter_datasets(records, datasets):
    wanted_raw = {name.strip() for name in datasets.split(",") if name.strip()}
    if not wanted_raw:
        return records
    # 别名映射：旧命令中 archive/pascal → pascal_archive
    alias_map = {
        "archive": "pascal_archive",
        "pascal": "pascal_archive",
        "pascal_archive": "pascal_archive",
        "physio": "physionet2016",
        "physionet": "physionet2016",
        "physionet2016": "physionet2016",
        "circor": "circor",
    }
    wanted = set()
    for name in wanted_raw:
        resolved = alias_map.get(name, name)
        wanted.add(resolved)
        if resolved != name:
            wanted.add(name)
    return [rec for rec in records if rec["dataset"] in wanted]


def bandpass_filter(x, sr, low=LOWCUT, high=HIGHCUT, order=5):
    nyq = 0.5 * sr
    high = min(high, nyq * 0.95)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x.astype(np.float32)
    return filtfilt(b, a, x).astype(np.float32)


def active_clip(x, sr, max_duration):
    if max_duration is None or max_duration <= 0:
        return x
    max_len = int(max_duration * sr)
    if len(x) <= max_len:
        return x

    win = int(2.0 * sr)
    hop = int(0.5 * sr)
    if len(x) <= win:
        return x[:max_len]
    best_start = 0
    best_energy = -1.0
    for start in range(0, max(1, len(x) - max_len), hop):
        seg = x[start : start + max_len]
        energy = float(np.mean(seg**2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return x[best_start : best_start + max_len]


def load_preprocessed_audio(path, target_sr, max_duration):
    y, sr = librosa.load(fs_path(path), sr=None, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = np.asarray(y, dtype=np.float32)
    y = y - float(np.mean(y))
    y = active_clip(y, sr, max_duration)
    y = bandpass_filter(y, sr)
    scale = np.percentile(np.abs(y), 99)
    if scale > 1e-8:
        y = y / scale
    return y.astype(np.float32), sr

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2) + 1e-12))


def snr_db(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    noise = reference - estimate
    return float(10.0 * np.log10((np.sum(reference**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def rmse(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((reference - estimate) ** 2) + 1e-12))


def sigmoid(x):
    x = np.clip(float(x), -40.0, 40.0)
    return float(1.0 / (1.0 + np.exp(-x)))


# --- Sensitivity-analysis hooks: defaults == frozen paper values; behavior is
# --- identical unless a caller overrides these module globals before invocation.
MIX_KAPPA_O = 16.0
MIX_NU0 = 0.55
MIX_KAPPA_R = 4.0
MIX_S0 = 0.90
MIX_LAMBDA0 = 0.25


def scale_noise(reference, noise, target_snr_db):
    ref_rms = rms(reference)
    noise_rms = rms(noise)
    target_noise_rms = ref_rms / (10.0 ** (target_snr_db / 20.0))
    return noise * (target_noise_rms / (noise_rms + 1e-12))


def bandlimited_noise(n, sr, rng):
    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    return bandpass_filter(noise, sr, low=30.0, high=350.0, order=4)


def white_noise(n, rng):
    return rng.normal(0.0, 1.0, n).astype(np.float32)


def pink_noise(n, rng):
    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    y = np.fft.irfft(spectrum * scale, n=n)
    y = y - np.mean(y)
    return y.astype(np.float32)


def parse_noise_scenario(scenario):
    scenario = scenario.strip().lower()
    if scenario in {"gwn_0db", "mixed_5db"}:
        return scenario, None
    for prefix in ("awgn_", "gwn_paper_", "apgn_", "pink_"):
        if scenario.startswith(prefix) and scenario.endswith("db"):
            snr_text = scenario[len(prefix) : -2]
            return prefix.rstrip("_"), float(snr_text)
    raise ValueError(f"unknown noise scenario: {scenario}")


def add_noise(clean, sr, scenario, rng):
    kind, target_snr = parse_noise_scenario(scenario)
    if kind in {"awgn", "gwn_paper"}:
        noise = white_noise(len(clean), rng)
        return clean + scale_noise(clean, noise, target_snr), target_snr

    if kind in {"apgn", "pink"}:
        noise = pink_noise(len(clean), rng)
        return clean + scale_noise(clean, noise, target_snr), target_snr

    if scenario == "gwn_0db":
        noise = bandlimited_noise(len(clean), sr, rng)
        return clean + scale_noise(clean, noise, 0.0), 0.0

    if scenario == "mixed_5db":
        n = len(clean)
        t = np.arange(n) / sr
        noise = 0.75 * bandlimited_noise(n, sr, rng)
        noise += 0.25 * np.sin(2 * np.pi * 32.0 * t + rng.uniform(0, 2 * np.pi))
        for _ in range(max(2, int(len(clean) / sr))):
            center = rng.integers(0, n)
            width = int(rng.integers(max(4, sr // 250), max(8, sr // 80)))
            a = max(0, center - width)
            b = min(n, center + width)
            if b > a:
                noise[a:b] += rng.uniform(-4.0, 4.0) * np.hanning(b - a)
        noise = bandpass_filter(noise, sr, low=25.0, high=400.0, order=4)
        return clean + scale_noise(clean, noise, 5.0), 5.0

    raise ValueError(f"unknown noise scenario: {scenario}")


def detect_cycles(pcg, sr, segmenter="existing"):
    segmenter = (segmenter or "existing").lower()
    if segmenter in {"existing", "phase_lock", "old"}:
        return detect_s1_s2(pcg, sr)

    if segmenter in {"v3", "xintiao_v3", "hybrid_v3"}:
        from xintiaonew import HybridHeartSoundDetector

        with contextlib.redirect_stdout(io.StringIO()):
            detector = HybridHeartSoundDetector(audio_data=pcg, target_sr=sr)
            s1, s2, used_bpm = detector.run()
        s1 = np.asarray(s1, dtype=int)
        s2 = np.asarray(s2, dtype=int)
        peaks = np.sort(np.concatenate([s1, s2])) if len(s1) + len(s2) > 0 else np.array([], dtype=int)
        t0 = 60.0 / float(used_bpm) if used_bpm and math.isfinite(float(used_bpm)) else math.nan
        return detector.env_combined, peaks, s1, s2, t0

    raise ValueError(f"unknown segmenter: {segmenter}")


def segmentation_pre_denoise(x, sr, mode="none"):
    mode = (mode or "none").lower()
    if mode in {"", "none", "raw"}:
        return x.astype(np.float32), {"mode": "none", "elapsed_sec": 0.0}
    raise ValueError("standalone AdaptiveV4 only supports segmentation_pre_denoise_mode='none'")


def _phase_aware_gate(y, s1_idx, s2_idx, sr, floor=0.3):
    """周期相位感知时域门控: 压低舒张期深处的纯静默底噪, 完整保留收缩期/S1-S2活跃区
    以及舒张期高能量(可能的 diastolic murmur)。floor>=1.0 等价于关闭门控(用于消融)。
    依据 (2026-06-29 smoke): 在有间隙样本上降底噪并小幅升 ΔSNR, 比纯能量门控更保杂音。"""
    if floor >= 1.0:
        return np.asarray(y, dtype=np.float32)
    n = len(y)
    yv = np.asarray(y, dtype=np.float64)
    s1 = np.asarray(s1_idx, dtype=int)
    s2 = np.asarray(s2_idx, dtype=int)
    margin = int(round(0.06 * sr))
    active = np.zeros(n, dtype=np.float64)
    for idx in np.concatenate([s1, s2]) if (len(s1) + len(s2)) else np.array([], dtype=int):
        if 0 <= idx < n:
            active[max(0, idx - margin):min(n, idx + margin)] = 1.0
    for a in s1:
        cand = s2[s2 > a]
        if len(cand) == 0:
            continue
        b = int(cand[0])
        if 0.05 <= (b - a) / float(sr) <= 0.60:
            active[max(0, a):min(n, b)] = 1.0
    w = max(1, int(0.03 * sr))
    active = np.clip(np.convolve(active, np.ones(w) / w, mode="same"), 0.0, 1.0)
    env = np.abs(hilbert(yv))
    we = max(1, int(0.04 * sr))
    env = np.convolve(env, np.ones(we) / we, mode="same")
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)) + 1e-9)
    env_high = 1.0 / (1.0 + np.exp(-(env - (med + 0.5 * mad)) / (1.5 * mad)))
    keep = np.clip(np.maximum(active, env_high), 0.0, 1.0)
    return (yv * (floor + (1.0 - floor) * keep)).astype(np.float32)


def run_baseline(
    clean,
    noisy,
    sr,
    periodic_params=None,
    algorithm_input_bandpass=False,
    segmenter="existing",
    segmentation_pre_denoise_mode="none",
    template_mode="mean",
    gate_floor=0.3,
):
    # template_mode: 精炼版默认 "mean"; 消融可传 "median" 回退到旧实现.
    # 依据 (2026-06-29 ablation smoke): mean 比 median 稳定高约 +0.09 dB ΔSNR,
    # 且在 worst-case/瞬态/murmur systole 中频保持上不劣于 median.
    periodic_params = periodic_params or {}
    start = time.perf_counter()
    noisy_for_algorithm = bandpass_filter(noisy, sr) if algorithm_input_bandpass else noisy
    cycle_input, pre_denoise_info = segmentation_pre_denoise(noisy_for_algorithm, sr, segmentation_pre_denoise_mode)
    env, peaks, s1, s2, t0 = detect_cycles(cycle_input, sr, segmenter=segmenter)
    if len(s1) < 4 or len(s2) < 3:
        # No usable quasi-periodic structure (severe corruption / arrhythmia):
        # fall back to the conservative band-pass baseline instead of failing,
        # so the degradation behaviour is well-defined for deployment.
        bp = np.asarray(noisy_for_algorithm, dtype=np.float32)
        return {
            "template": bp, "cycle_hf": bp, "periodic": bp, "periodic_v2": bp,
            "adaptive_v3": bp, "resid": np.zeros_like(bp),
            "snr_algorithm_input_db": snr_db(clean, noisy_for_algorithm),
            "snr_segmentation_input_db": snr_db(clean, cycle_input),
            "s1_count": int(len(s1)), "s2_count": int(len(s2)),
            "s1_times_sec": np.asarray(s1, dtype=np.float64) / float(sr),
            "s2_times_sec": np.asarray(s2, dtype=np.float64) / float(sr),
            "t0_sec": float(t0) if t0 is not None else math.nan,
            "segmenter": segmenter,
            "segmentation_pre_denoise": pre_denoise_info.get("mode", "none"),
            "segmentation_pre_denoise_elapsed_sec": pre_denoise_info.get("elapsed_sec", 0.0),
            "hf_decision": "fallback_no_cycle", "hf_sys_ratio": math.nan, "hf_dia_ratio": math.nan,
            "periodic_decision": "fallback_no_cycle", "periodic_mask_ratio": math.nan,
            "periodic_added_rms_ratio": math.nan, "periodic_ck_mean": math.nan,
            "periodic_energy_ratio_p90": math.nan,
            "periodic_v2_decision": "fallback_no_cycle", "periodic_v2_gate_mean": math.nan,
            "periodic_v2_added_rms_ratio": math.nan, "periodic_v2_alpha_eff": 0.0,
            "periodic_v2_concentration": math.nan, "periodic_v2_freq_strength": math.nan,
            "periodic_v2_cycle_confidence": math.nan, "periodic_v2_residual_strength": math.nan,
            "outband_noise_ratio": math.nan, "adaptive_v3_outband_gate": math.nan,
            "adaptive_v3_residual_gate": math.nan, "adaptive_v3_weight": 0.0,
            "fallback": "bandpass_no_cycle",
            "elapsed_sec": time.perf_counter() - start,
        }

    x_hat, resid, s1_ref, s2_ref, _, _ = denoise_pcg_by_s1s2_templates(
        noisy_for_algorithm,
        sr,
        s1,
        s2,
        win_ms=80,
        refine_ms=80,
        template_mode=template_mode,
        iters=2,
    )
    if x_hat is None:
        raise RuntimeError("template reconstruction failed")

    hf, hf_info = recover_hf_by_phase_dominance_with_freq_consistency(
        resid,
        sr,
        s1_ref,
        s2_ref,
        f_low=100,
        f_high=300,
        dominance_ratio=0.9,
        exclude_ms=20,
        freq_tol=50,
    )
    x_cycle = (x_hat + hf).astype(np.float32)
    x_periodic, periodic_info = recover_periodic_residual(
        x_hat,
        resid,
        sr,
        s1_ref,
        f_low=periodic_params.get("f_low", 80),
        f_high=periodic_params.get("f_high", 350),
        alpha=periodic_params.get("alpha", 0.45),
        keep_quantile=periodic_params.get("keep_quantile", 0.82),
        min_z=periodic_params.get("min_z", 0.5),
        max_added_rms_ratio=periodic_params.get("max_added_rms_ratio", 0.35),
    )
    x_periodic_v2, periodic_v2_info = recover_cycle_consistent_residual(
        x_hat,
        resid,
        sr,
        s1_ref,
        f_low=periodic_params.get("f_low", DEFAULT_ADAPTIVE_V4_CONFIG.cycle_residual_band[0]),
        f_high=periodic_params.get("f_high", DEFAULT_ADAPTIVE_V4_CONFIG.cycle_residual_band[1]),
        alpha_max=periodic_params.get("v2_alpha_max", DEFAULT_ADAPTIVE_V4_CONFIG.gate_max),
        max_added_rms_ratio=periodic_params.get("v2_max_added_rms_ratio", 0.50),
    )
    residual_strength = periodic_v2_info.get("residual_strength", math.nan)
    outband_noise_ratio = rms(noisy - noisy_for_algorithm) / (rms(noisy_for_algorithm) + 1e-12)
    if algorithm_input_bandpass:
        # The band-pass input is already a strong high-SNR baseline under the
        # paper-aligned AWGN/APGN protocol. Keep adaptive residual recovery only
        # when the removed out-of-band noise is large enough to imply severe
        # corruption, and let residual evidence modulate but not dominate it.
        outband_weight = sigmoid(MIX_KAPPA_O * (outband_noise_ratio - MIX_NU0))
        if isinstance(residual_strength, float) and math.isfinite(residual_strength):
            residual_weight = sigmoid(MIX_KAPPA_R * (residual_strength - MIX_S0))
        else:
            residual_weight = 0.5
        adaptive_weight = float(np.clip(outband_weight * (MIX_LAMBDA0 + (1.0 - MIX_LAMBDA0) * residual_weight), 0.0, 1.0))
    elif isinstance(residual_strength, float) and math.isfinite(residual_strength):
        residual_weight = sigmoid(5.0 * (residual_strength - 1.0))
        outband_weight = math.nan
        adaptive_weight = residual_weight
    else:
        outband_weight = math.nan
        residual_weight = math.nan
        adaptive_weight = 0.65
    x_adaptive = (adaptive_weight * x_periodic_v2 + (1.0 - adaptive_weight) * noisy_for_algorithm).astype(np.float32)
    # 相位感知时域门控: 压低舒张期静默段底噪, 保留心音/杂音活跃区 (gate_floor>=1.0 关闭)
    x_adaptive = _phase_aware_gate(x_adaptive, s1_ref, s2_ref, sr, floor=gate_floor)
    elapsed = time.perf_counter() - start

    s1_times_sec = np.asarray(s1_ref, dtype=np.float64) / float(sr) if len(s1_ref) else np.array([], dtype=np.float64)
    s2_times_sec = np.asarray(s2_ref, dtype=np.float64) / float(sr) if len(s2_ref) else np.array([], dtype=np.float64)
    return {
        "template": x_hat,
        "cycle_hf": x_cycle,
        "periodic": x_periodic,
        "periodic_v2": x_periodic_v2,
        "adaptive_v3": x_adaptive,
        "resid": resid,
        "snr_algorithm_input_db": snr_db(clean, noisy_for_algorithm),
        "snr_segmentation_input_db": snr_db(clean, cycle_input),
        "s1_count": int(len(s1_ref)),
        "s2_count": int(len(s2_ref)),
        "s1_times_sec": s1_times_sec,
        "s2_times_sec": s2_times_sec,
        "t0_sec": float(t0) if t0 is not None else math.nan,
        "segmenter": segmenter,
        "segmentation_pre_denoise": pre_denoise_info.get("mode", "none"),
        "segmentation_pre_denoise_elapsed_sec": pre_denoise_info.get("elapsed_sec", 0.0),
        "hf_decision": hf_info.get("decision", ""),
        "hf_sys_ratio": hf_info.get("sys_ratio", math.nan),
        "hf_dia_ratio": hf_info.get("dia_ratio", math.nan),
        "periodic_decision": periodic_info.get("decision", ""),
        "periodic_mask_ratio": periodic_info.get("mask_ratio", math.nan),
        "periodic_added_rms_ratio": periodic_info.get("added_rms_ratio", math.nan),
        "periodic_ck_mean": periodic_info.get("correlated_kurtosis_mean", math.nan),
        "periodic_energy_ratio_p90": periodic_info.get("energy_ratio_p90", math.nan),
        "periodic_v2_decision": periodic_v2_info.get("decision", ""),
        "periodic_v2_gate_mean": periodic_v2_info.get("gate_mean", math.nan),
        "periodic_v2_added_rms_ratio": periodic_v2_info.get("added_rms_ratio", math.nan),
        "periodic_v2_alpha_eff": periodic_v2_info.get("alpha_eff", math.nan),
        "periodic_v2_concentration": periodic_v2_info.get("concentration", math.nan),
        "periodic_v2_freq_strength": periodic_v2_info.get("freq_strength", math.nan),
        "periodic_v2_cycle_confidence": periodic_v2_info.get("cycle_confidence", math.nan),
        "periodic_v2_residual_strength": periodic_v2_info.get("residual_strength", math.nan),
        "outband_noise_ratio": outband_noise_ratio,
        "adaptive_v3_outband_gate": outband_weight,
        "adaptive_v3_residual_gate": residual_weight,
        "adaptive_v3_weight": adaptive_weight,
        "elapsed_sec": elapsed,
    }


def clean_json_value(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# AdaptiveV4 diagnostic-risk fusion
# ============================================================


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


MANIFEST_PATH = PROJECT_ROOT / "datasets" / "heart_sound_unified" / "manifests" / "paper_aligned_recordings.csv"
NOISE_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "noise_manifests" / "noise_manifest.csv"


def read_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def filter_by_dataset(records: list[dict], datasets: str) -> list[dict]:
    wanted = {name.strip() for name in datasets.split(",") if name.strip()}
    if not wanted:
        return records
    aliases = {
        "archive": "pascal_archive",
        "pascal": "pascal_archive",
        "physio": "physionet2016",
        "physionet": "physionet2016",
    }
    resolved = {aliases.get(name, name) for name in wanted}
    return [r for r in records if r.get("dataset") in resolved]


def stable_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def split_rows(records: list[dict], split: str, train_percent: int) -> list[dict]:
    split = split.lower()
    out = []
    for row in records:
        key = f"{row.get('dataset', '')}::{row.get('record_id', '')}"
        in_train = stable_int(key) % 100 < train_percent
        if (split == "train" and in_train) or (split == "eval" and not in_train):
            out.append(row)
    return out


def load_noise_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("usable") == "1"]


def group_noise_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("noise_category", "")].append(row)
    return dict(grouped)


def parse_real_scenario(scenario: str) -> tuple[str, float]:
    scenario = scenario.strip().lower()
    if not scenario.startswith("real_") or not scenario.endswith("db"):
        raise ValueError(f"invalid real noise scenario: {scenario}")
    body = scenario[len("real_"):-len("db")]
    category, snr_text = body.rsplit("_", 1)
    return category, float(snr_text)


def pick_noise(record_id: str, scenario: str, category: str, candidates: list[dict], seed: int) -> dict:
    if not candidates:
        raise ValueError(f"no usable noise candidates for category '{category}'")
    idx = stable_int(f"{record_id}::{scenario}::{category}::{seed}") % len(candidates)
    return candidates[idx]


def load_real_noise(noise_entry: dict, target_sr: int) -> np.ndarray:
    y, sr = librosa.load(noise_entry["path"], sr=None, mono=True)
    if y.size == 0:
        raise ValueError(f"empty noise audio: {noise_entry['path']}")
    y = np.asarray(y, dtype=np.float64)
    y = y - float(np.mean(y))
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return np.asarray(y, dtype=np.float64)


def align_noise(noise: np.ndarray, clean_len: int, record_id: str, scenario: str, category: str, seed: int) -> np.ndarray:
    if len(noise) >= clean_len:
        start = stable_int(f"{record_id}::{scenario}::{category}::{seed}::crop") % max(1, len(noise) - clean_len + 1)
        return noise[start:start + clean_len]
    repeats = int(np.ceil(clean_len / max(1, len(noise))))
    return np.tile(noise, repeats)[:clean_len]


def mix_real_noise(clean: np.ndarray, sr: int, record_id: str, scenario: str, noise_entry: dict, seed: int) -> tuple[np.ndarray, dict, float]:
    category, target_snr = parse_real_scenario(scenario)
    noise = load_real_noise(noise_entry, sr)
    noise = align_noise(noise, len(clean), record_id, scenario, category, seed)
    noise = scale_noise(clean, noise, target_snr)
    noisy = clean.astype(np.float64) + noise.astype(np.float64)
    return noisy.astype(np.float32), noise_entry, snr_db(clean, noisy)


def make_noisy(
    clean: np.ndarray,
    sr: int,
    scenario: str,
    rng: np.random.Generator,
    noise_groups: dict[str, list[dict]] | None,
    record_id: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if scenario.strip().lower().startswith("real_"):
        if noise_groups is None:
            raise ValueError("real noise scenario requested without noise manifest")
        category, target_snr = parse_real_scenario(scenario)
        entry = pick_noise(record_id, scenario, category, noise_groups.get(category, []), seed)
        noisy, noise_entry, input_snr = mix_real_noise(clean, sr, record_id, scenario, entry, seed)
        return noisy, {
            "noise_type": scenario,
            "noise_category": category,
            "noise_id": noise_entry.get("noise_id", ""),
            "noise_path": noise_entry.get("path", ""),
            "target_snr_db": target_snr,
            "input_snr_db": input_snr,
        }

    noisy, target_snr = add_noise(clean, sr, scenario, rng)
    return noisy.astype(np.float32), {
        "noise_type": scenario,
        "noise_category": "",
        "noise_id": "",
        "noise_path": "",
        "target_snr_db": target_snr,
        "input_snr_db": snr_db(clean, noisy),
    }


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference = reference[:n] - float(np.mean(reference[:n]))
    estimate = estimate[:n] - float(np.mean(estimate[:n]))
    alpha = float(np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12))
    target = alpha * reference
    error = estimate - target
    return float(10.0 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(error ** 2) + 1e-12)))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    return float(np.sqrt(np.mean(err ** 2) + 1e-12))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def prd(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    return float(100.0 * np.linalg.norm(err) / (np.linalg.norm(reference) + 1e-12))


def pearson_corr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    if n < 2:
        return math.nan
    return float(np.corrcoef(reference[:n], estimate[:n])[0, 1])


def spectral_metrics(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    n = min(len(reference), len(estimate))
    ref_mag = np.abs(np.fft.rfft(np.asarray(reference[:n], dtype=np.float64)))
    est_mag = np.abs(np.fft.rfft(np.asarray(estimate[:n], dtype=np.float64)))
    log_ref = 20.0 * np.log10(ref_mag + 1e-8)
    log_est = 20.0 * np.log10(est_mag + 1e-8)
    lsd = float(np.sqrt(np.mean((log_ref - log_est) ** 2)))
    sc = float(np.linalg.norm(est_mag - ref_mag) / (np.linalg.norm(ref_mag) + 1e-12))
    return lsd, sc


def band_energy(x: np.ndarray, sr: int, low: float, high: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    spec = np.fft.rfft(x)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(np.abs(spec[mask]) ** 2) / max(1, len(x)))


def band_retention(reference: np.ndarray, estimate: np.ndarray, sr: int, low: float, high: float) -> float:
    return float(band_energy(estimate, sr, low, high) / (band_energy(reference, sr, low, high) + 1e-12))


def clipping_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0


def silent_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.abs(x) <= 1e-5)) if len(x) else 0.0


def compute_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, sr: int) -> dict:
    denoised = fit_length(denoised, len(clean)).astype(np.float32)
    input_snr = snr_db(clean, noisy)
    output_snr = snr_db(clean, denoised)
    input_si = si_sdr_db(clean, noisy)
    output_si = si_sdr_db(clean, denoised)
    error = np.asarray(clean, dtype=np.float64) - np.asarray(denoised, dtype=np.float64)
    input_error = np.asarray(clean, dtype=np.float64) - np.asarray(noisy, dtype=np.float64)
    rmse_val = rmse(clean, denoised)
    lsd, sc = spectral_metrics(clean, denoised)
    return {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "input_si_sdr_db": input_si,
        "output_si_sdr_db": output_si,
        "si_sdr_improvement_db": output_si - input_si,
        "rmse": rmse_val,
        "mae": mae(clean, denoised),
        "mse": float(np.mean(error ** 2)),
        "prd": prd(clean, denoised),
        "nrmse": float(rmse_val / (rms(clean) + 1e-12)),
        "pearson_corr": pearson_corr(clean, denoised),
        "log_spectral_distance": lsd,
        "spectral_convergence": sc,
        "heart_band_retention_20_150": band_retention(clean, denoised, sr, 20.0, 150.0),
        "mid_band_retention_150_400": band_retention(clean, denoised, sr, 150.0, 400.0),
        "high_band_retention_400_800": band_retention(clean, denoised, sr, 400.0, 800.0),
        "noise_reduction_db": float(10.0 * np.log10((np.sum(input_error ** 2) + 1e-12) / (np.sum(error ** 2) + 1e-12))),
        "residual_energy_ratio": float((np.sum(error ** 2) + 1e-12) / (np.sum(input_error ** 2) + 1e-12)),
        "duration_sec": float(len(clean) / sr),
        "clipping_ratio": clipping_ratio(denoised),
        "silent_ratio": silent_ratio(denoised),
    }


# ------------------------------------------------------------------
# 已删除未使用的 V4 诊断风险融合层与独立 CLI(band-guard/simplex/diagnostic_risk_fusion/
# build_variants/run/__main__ 及 DIAGNOSTIC_RISK_BANDS/METHODS/FIELDS)。
# 论文方法 ACPD = run_baseline(...)['adaptive_v3'], 见上。2026-07-03 对齐论文精简。
# ------------------------------------------------------------------
