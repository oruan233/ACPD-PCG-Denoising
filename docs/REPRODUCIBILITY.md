# Reproducing ACPD

Install the package and run the data-free smoke tests before processing a large
collection. The default is the data-adaptive JBHI main-table implementation.
See [VERSIONS.md](VERSIONS.md) for the exact source trace and the fixed version.

```bash
pip install -e .
python tests/test_smoke.py
python tests/test_release.py
```

## Reference recordings

Supply your own WAV files under `data/reference_wavs`. A directory run identifies
each recording by its path relative to that directory, without the extension.
For exact paper-style IDs or multiple datasets, use a CSV with columns
`dataset,record_id,audio_path` and pass `--record-manifest`. Relative audio paths
resolve from the manifest's directory. Duplicate `(dataset, record_id)` entries
are rejected. The script uses precisely the selected files and does not infer
train/evaluation splits from a directory name.

References undergo the original research preprocessing: mono conversion,
resampling to 2 kHz, mean removal, active-region cropping, 25-400 Hz band-pass
and amplitude-quantile normalization. No evaluation-only band-pass is applied
to method outputs. Recordings shorter than 4 seconds after preprocessing are
excluded by default.

## Synthetic noise

```bash
python scripts/evaluate.py --data-root data/reference_wavs \
  --output-dir outputs/synthetic --noise-types awgn apgn \
  --snr-db -6 -3 0 3 6 --seeds 20260515 20260516 20260517
```

`pink` is also accepted as the research generator's APGN alias. Specify scenarios
explicitly when reproducing a historical run; a default config does not promise
an exact historical table selection.

## Real recorded noise

Prepare a UTF-8 noise-manifest CSV with columns
`noise_id,noise_category,path,usable`. `usable` must be `1`; other rows are ignored.
Use categories `lung_respiratory`, `ambient`, `speech`, `cough`, `crumpling`
or your own category names. Paths can be absolute or relative to the manifest.
Keep IDs and row ordering stable for exact mixtures.

```csv
noise_id,noise_category,path,usable
clip_001,speech,../noise/speech_001.wav,1
```

```bash
python scripts/evaluate.py --data-root data/reference_wavs \
  --noise-manifest data/noise_manifests/noise_manifest.csv \
  --noise-categories speech cough --snr-db -6 0 6 \
  --output-dir outputs/real
```

Real-noise clips undergo the original mono/demean/resample processing, followed
by deterministic crop or repetition and target-SNR scaling. They are added to
reference recordings. This evaluates controlled additive mixtures, rather than
naturally contaminated clinical recordings.

## Config dispatcher and cumulative ablation

```bash
python scripts/reproduce_results.py --experiment synthetic_noise \
  --data-root data/reference_wavs --output-dir outputs/synthetic
python scripts/reproduce_results.py --experiment real_noise \
  --data-root data/reference_wavs \
  --noise-manifest data/noise_manifests/noise_manifest.csv --output-dir outputs/real
python scripts/reproduce_results.py --experiment ablation \
  --data-root data/reference_wavs --output-dir outputs/ablation
```

The four cumulative stages are template only, template plus residual reinjection,
adaptive fusion without final phase gating, and full ACPD. They use the same
mixture and intermediate reconstruction. Select the historical method with
`--version fixed`. This produces a new ablation on your inputs; it does not claim
to regenerate the published ablation without its exact original cohort/settings.

`configs/default.yaml` documents method conventions; it is not read as a runtime
parameter override. The other configs are read by the dispatcher.

## Randomness, aggregation and output

The mixture seed uses the research runner's MD5 mapping of
`record_id::scenario::seed`, taking its first eight hex digits. The denoiser itself
has no stochastic training/inference. The default experiment seed is `20260515`.

- `per_record_metrics.csv`: one row per recording/scenario/seed/variant.
- `recording_metrics.csv`: average across scenarios/seeds within each recording.
- `overall_metrics.csv`: per-dataset averages and bootstrap 95% intervals across
  those recording means. Each original recording has equal weight.
- `aggregate_metrics.csv`: per-scenario recording-level averages and intervals.
- `run_summary.json`: candidate counts, seeds, scenarios, skipped/failed counts.
- `errors.jsonl`: errors written immediately. Failed runs return a nonzero exit
  code and retain any successfully computed metrics for inspection.

Main metrics are `snr_improvement_db`, `si_sdr_improvement_db`, `rmse`, and `mae`.
RMSE/MAE are dimensionless on the normalized references. Bootstrap intervals use
1000 resamples by default; change `--bootstrap-samples` for a chosen analysis.

Run with `--limit 1` first. Do not present a partial run with failures as a complete
comparison. Exact historical main-table reproduction also requires the original
cohort, record IDs, manifest ordering, scenarios and seeds. Baseline comparison
models and clinical outcome validation are outside this algorithm release.
