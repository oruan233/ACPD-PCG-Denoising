#!/usr/bin/env python3
"""Evaluate ACPD versions or cumulative modules on supplied reference WAVs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acpd
from acpd import adaptive_v4 as core

METRICS = ('snr_improvement_db', 'si_sdr_improvement_db', 'rmse', 'mae')
ABLATIONS = ('template_only', 'residual_reinjection', 'adaptive_fusion', 'full')


def _seed_for(record: str, scenario: str, base_seed: int) -> int:
    """Use the MD5 seed mapping from the paper-table runner."""
    return int(hashlib.md5(f'{record}::{scenario}::{base_seed}'.encode('utf-8')).hexdigest()[:8], 16)


def build_scenarios(noise_types, snrs):
    if not snrs or not all(np.isfinite(s) for s in snrs):
        raise ValueError('SNR values must be finite')
    return [f'{noise}_{snr:g}db' for noise in noise_types for snr in snrs]


def _records(data_root, record_manifest=None):
    if record_manifest:
        manifest = Path(record_manifest).resolve()
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        records = []
        for row in rows:
            audio_path = Path(row['audio_path'])
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            records.append((row['dataset'], row['record_id'], audio_path))
    else:
        data_root = Path(data_root).resolve()
        records = [('user', p.relative_to(data_root).with_suffix('').as_posix(), p)
                   for p in sorted(data_root.rglob('*')) if p.suffix.lower() == '.wav']
    identities = [(dataset, record) for dataset, record, _path in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate (dataset, record_id) entries in the record manifest')
    return records


def _noise_groups(manifest_path):
    path = Path(manifest_path).resolve()
    rows = core.load_noise_manifest(path)
    for row in rows:
        audio = Path(row['path'])
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            raise ValueError(f'Noise clip does not exist: {audio}')
        row['path'] = str(audio)
    return core.group_noise_by_category(rows)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _means(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    return [{**dict(zip(fields, key)), 'observations': len(items),
             **{metric: float(np.mean([item[metric] for item in items])) for metric in METRICS}}
            for key, items in groups.items()]


def _aggregates(record_rows, fields, bootstrap_samples, rng):
    groups = defaultdict(list)
    for row in record_rows:
        groups[tuple(row[f] for f in fields)].append(row)
    output = []
    for key, items in groups.items():
        result = {**dict(zip(fields, key)), 'n_records': len(items)}
        for metric in METRICS:
            values = np.array([item[metric] for item in items])
            draws = np.array([rng.choice(values, len(values), replace=True).mean()
                              for _ in range(bootstrap_samples)])
            low, high = np.quantile(draws, [0.025, 0.975])
            result.update({metric: float(values.mean()), metric + '_ci_low': float(low),
                           metric + '_ci_high': float(high), metric + '_ci_halfwidth': float((high - low) / 2)})
        output.append(result)
    return output


def evaluate(data_root, output_dir, scenarios, base_seed=20260515, min_duration=4.0,
             limit=0, *, version='adaptive', noise_manifest=None, record_manifest=None,
             seeds=None, ablation=False, bootstrap_samples=1000):
    if min_duration < 0 or limit < 0 or bootstrap_samples < 1:
        raise ValueError('min_duration/limit must be nonnegative and bootstrap_samples positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear diagnostics from a previous run; append each failure immediately.
    error_log = output_dir / 'errors.jsonl'
    error_log.write_text('', encoding='utf-8')
    failures = 0
    skipped = 0
    def log_error(record, scenario, exc):
        nonlocal failures
        failures += 1
        with error_log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'record': record, 'scenario': scenario, 'error': str(exc)},
                                    ensure_ascii=False) + '\n')

    records = _records(data_root, record_manifest)
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError('No reference WAV files were found')
    real = [s for s in scenarios if s.startswith('real_')]
    groups = _noise_groups(noise_manifest) if noise_manifest else None
    if real and groups is None:
        raise ValueError('--noise-manifest is required for real-noise scenarios')
    for scenario in real:
        category, _snr = core.parse_real_scenario(scenario)
        if not groups.get(category):
            raise ValueError(f'No usable noise clips for category {category!r}')
    rows = []
    variants = ABLATIONS if ablation else ('full',)
    for index, (dataset, record_id, path) in enumerate(records):
        try:
            clean, sr = core.load_preprocessed_audio(path, core.TARGET_SR, 0.0)
            if clean.size < min_duration * sr:
                skipped += 1
                continue
        except Exception as exc:
            log_error(record_id, 'load', exc)
            continue
        for seed in seeds or [base_seed]:
            for scenario in scenarios:
                try:
                    rng = np.random.default_rng(_seed_for(record_id, scenario, seed))
                    if scenario.startswith('real_'):
                        noisy, metadata = core.make_noisy(clean, sr, scenario, rng, groups, record_id, seed)
                    else:
                        noisy, _snr = core.add_noise(clean, sr, scenario, rng)
                        metadata = {}
                    result = acpd.run_baseline(clean, noisy, sr, version=version,
                                               gate_floor=1.0 if ablation else 0.3)
                    outputs = {'full': result['adaptive_v3']}
                    if ablation:
                        # This follows the original cumulative module runner:
                        # template -> residual candidate -> fusion -> phase gate.
                        s1 = np.rint(np.asarray(result['s1_times_sec']) * sr).astype(int)
                        s2 = np.rint(np.asarray(result['s2_times_sec']) * sr).astype(int)
                        outputs = {'template_only': result['template'],
                                   'residual_reinjection': result['periodic_v2'],
                                   'adaptive_fusion': result['adaptive_v3'],
                                   'full': (result['adaptive_v3'] if version == 'fixed' and result.get('fallback')
                                            else core._phase_aware_gate(result['adaptive_v3'], s1, s2, sr))}
                    for variant in variants:
                        metrics = core.compute_metrics(clean, noisy, outputs[variant], sr)
                        selected = {key: float(metrics[key]) for key in METRICS}
                        if not all(np.isfinite(value) for value in selected.values()):
                            raise ValueError('Nonfinite evaluation metric')
                        rows.append({'dataset': dataset, 'record': record_id, 'scenario': scenario,
                                     'seed': seed, 'version': version, 'variant': variant,
                                     'noise_id': metadata.get('noise_id', ''),
                                     'input_snr_db': metrics['input_snr_db'], **selected})
                except Exception as exc:
                    log_error(record_id, scenario, exc)
        print(f'[{index + 1}/{len(records)}] {record_id}', flush=True)
    summary = {'version': version, 'candidate_records': len(records), 'observations': len(rows),
               'skipped_short': skipped, 'failures': failures, 'seeds': seeds or [base_seed],
               'scenarios': scenarios, 'ablation': ablation, 'bootstrap_samples': bootstrap_samples}
    (output_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not rows:
        raise RuntimeError(f'No observations evaluated; see {error_log}')
    _write_csv(output_dir / 'per_record_metrics.csv', rows)
    record_rows = _means(rows, ['dataset', 'record', 'version', 'variant'])
    _write_csv(output_dir / 'recording_metrics.csv', record_rows)
    rng = np.random.default_rng(base_seed)
    _write_csv(output_dir / 'overall_metrics.csv',
               _aggregates(record_rows, ['dataset', 'version', 'variant'], bootstrap_samples, rng))
    scenario_records = _means(rows, ['dataset', 'record', 'scenario', 'version', 'variant'])
    _write_csv(output_dir / 'aggregate_metrics.csv',
               _aggregates(scenario_records, ['dataset', 'scenario', 'version', 'variant'], bootstrap_samples, rng))
    if failures:
        raise RuntimeError(f'{failures} observations failed; partial outputs saved, see {error_log}')
    return output_dir / 'per_record_metrics.csv', output_dir / 'aggregate_metrics.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/reference_wavs')
    parser.add_argument('--record-manifest')
    parser.add_argument('--output-dir', default='outputs/evaluation')
    parser.add_argument('--noise-types', nargs='+', choices=['awgn', 'apgn', 'pink'], default=['awgn', 'apgn'])
    parser.add_argument('--snr-db', nargs='+', type=float, default=[-6, -3, 0, 3, 6])
    parser.add_argument('--noise-manifest')
    parser.add_argument('--noise-categories', nargs='+', default=['ambient', 'lung_respiratory', 'speech', 'cough', 'crumpling'])
    parser.add_argument('--seed', type=int, default=20260515)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--min-duration', type=float, default=4.0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--version', choices=acpd.VERSIONS, default='adaptive')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    args = parser.parse_args()
    types = ['real_' + category for category in args.noise_categories] if args.noise_manifest else args.noise_types
    evaluate(args.data_root, args.output_dir, build_scenarios(types, args.snr_db), args.seed,
             args.min_duration, args.limit, version=args.version, noise_manifest=args.noise_manifest,
             record_manifest=args.record_manifest, seeds=args.seeds, ablation=args.ablation,
             bootstrap_samples=args.bootstrap_samples)


if __name__ == '__main__':
    main()
