#!/usr/bin/env python3
"""Run synthetic noise, real recorded noise, or cumulative module ablation."""
import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate import build_scenarios, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['synthetic_noise', 'real_noise', 'ablation'], required=True)
    parser.add_argument('--data-root', default='data/reference_wavs')
    parser.add_argument('--record-manifest')
    parser.add_argument('--noise-manifest')
    parser.add_argument('--output-dir', default='outputs/reproduce')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--version', choices=['adaptive', 'fixed'], default='adaptive')
    args = parser.parse_args()
    with (ROOT / 'configs' / f'{args.experiment}.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    noise = config['noise']
    real = args.experiment == 'real_noise'
    if real and not args.noise_manifest:
        parser.error('real_noise requires --noise-manifest; see docs/REPRODUCIBILITY.md')
    types = ['real_' + category for category in noise['categories']] if real else noise['types']
    evaluate(args.data_root, args.output_dir, build_scenarios(types, noise['input_snr_db']),
             config['seed'], config['min_duration_sec'], args.limit, version=args.version,
             noise_manifest=args.noise_manifest if real else None,
             record_manifest=args.record_manifest, seeds=config.get('seeds'),
             ablation=args.experiment == 'ablation')


if __name__ == '__main__':
    main()
