"""ACPD file/array inference wrapper.

CLI: python scripts/denoise.py --input noisy.wav --output denoised.wav
The versioned pipeline selects adaptive_full by default, with the frozen
fixed-parameter version available explicitly. The historical output dictionary
key 'adaptive_v3' is retained for compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


from .pipeline import run_baseline
from .adaptive_v4 import bandpass_filter


DEFAULT_METHOD = "adaptive_v3"  # = ACPD (paper method)
# Legacy names kept for backward compatibility; all map to the paper method.
METHOD_ALIASES = {
    "acpd": DEFAULT_METHOD,
    "adaptive_v4": DEFAULT_METHOD,
    "our_adaptive_v4": DEFAULT_METHOD,
    "adaptive_v4_diagnostic_risk_fusion": DEFAULT_METHOD,
}
_INFO_KEYS = ("adaptive_v3_weight", "outband_noise_ratio", "s1_count", "s2_count",
              "periodic_v2_decision", "fallback", "elapsed_sec")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def _normalize_if_needed(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return audio / peak
    return audio


def _jsonable(v):
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.integer):
        v = int(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def denoise_array(noisy: np.ndarray, sr: int, method: str = DEFAULT_METHOD,
                  *, version: str = 'adaptive') -> tuple[np.ndarray, dict]:
    """Denoise a mono waveform with ACPD.

    At inference time only the noisy waveform is available; it is passed as the
    length/reference placeholder (``clean``), which run_baseline uses only for
    diagnostic SNR fields and never in the denoising path itself.
    """
    method = METHOD_ALIASES.get(method, method)
    if method != DEFAULT_METHOD:
        raise ValueError(f"Unknown method '{method}'. Only '{DEFAULT_METHOD}' (ACPD) is available.")
    if not isinstance(sr, (int, np.integer)) or sr < 800:
        raise ValueError('sample rate must be an integer >= 800 Hz')
    noisy = _normalize_if_needed(_to_mono(noisy))
    if noisy.size == 0 or not np.all(np.isfinite(noisy)):
        raise ValueError('audio must be nonempty and finite')
    input_length = len(noisy)
    noisy = noisy - float(np.mean(noisy))
    factor = math.gcd(int(sr), 2000)
    analysis = resample_poly(noisy, 2000 // factor, int(sr) // factor) if sr != 2000 else noisy
    if analysis.size < 32:
        raise ValueError('audio is too short for the analysis filters')
    try:
        result = run_baseline(analysis, analysis, 2000, version=version)
    except RuntimeError as exc:
        # The frozen detector raises on signals with no estimable cycle (e.g.
        # silence). Handle precisely that condition at the public wrapper.
        if str(exc) != 'T0 estimation failed':
            raise
        result = {'adaptive_v3': bandpass_filter(analysis, 2000),
                  'fallback': 'bandpass_no_cycle', 's1_count': 0, 's2_count': 0,
                  'adaptive_v3_weight': 0.0,
                  'parameter_mode': 'adaptive_full' if version == 'adaptive' else 'fixed'}
    denoised = np.asarray(result["adaptive_v3"], dtype=np.float64)
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != 2000:
        denoised = resample_poly(denoised, int(sr) // factor, 2000 // factor)
    denoised = denoised[:input_length]
    if len(denoised) < input_length:
        denoised = np.pad(denoised, (0, input_length - len(denoised)))
    denoised = _normalize_if_needed(denoised)
    info = {k: _jsonable(result.get(k)) for k in _INFO_KEYS}
    info.update(version=version, parameter_mode=result['parameter_mode'], analysis_sample_rate=2000)
    return denoised.astype(np.float32), info


def denoise_file(input_path: Path | str, output_path: Path | str, method: str = DEFAULT_METHOD,
                 *, version: str = 'adaptive') -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    audio, sr = sf.read(str(input_path), always_2d=False)
    denoised, info = denoise_array(audio, sr, method=method, version=version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), denoised, sr)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_rate": int(sr),
        "samples": int(len(denoised)),
        "method": DEFAULT_METHOD,
        "info": info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACPD standalone WAV denoiser.")
    parser.add_argument("--input", required=True, help="Input noisy WAV path.")
    parser.add_argument("--output", required=True, help="Output denoised WAV path.")
    parser.add_argument("--method", default=DEFAULT_METHOD,
                        help="Method name. Only 'adaptive_v3' (ACPD) is available.")
    parser.add_argument("--info-json", default="", help="Optional path to write run metadata JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = denoise_file(Path(args.input), Path(args.output), method=args.method)
    if args.info_json:
        info_path = Path(args.info_json)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
