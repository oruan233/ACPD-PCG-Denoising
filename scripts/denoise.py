#!/usr/bin/env python3
"""Denoise one or more heart-sound WAV files with ACPD (training-free).

Examples
--------
Single file:
    python scripts/denoise.py --input noisy.wav --output denoised.wav

Batch (directory of .wav files):
    python scripts/denoise.py --input path/to/wavs --output path/to/out_dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acpd import denoise_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Denoise heart-sound WAV(s) with ACPD (Adaptive Cycle-Prior Denoising)."
    )
    ap.add_argument("--input", required=True,
                    help="Input noisy WAV file, or a directory of WAV files.")
    ap.add_argument("--output", required=True,
                    help="Output WAV path (single input) or output directory (batch).")
    ap.add_argument("--info-json", default="",
                    help="Optional path to write run metadata as JSON.")
    ap.add_argument('--version', choices=['adaptive', 'fixed'], default='adaptive')
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    results = []
    if inp.is_dir():
        wavs = sorted(inp.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"No .wav files found in {inp}")
        out.mkdir(parents=True, exist_ok=True)
        for w in wavs:
            r = denoise_file(w, out / f"{w.stem}_denoised.wav", version=args.version)
            results.append(r)
            print(f"[ok] {w.name} -> {Path(r['output']).name}")
    else:
        results.append(denoise_file(inp, out, version=args.version))
        print(f"[ok] {inp.name} -> {out}")

    if args.info_json:
        p = Path(args.info_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
