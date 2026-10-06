# Algorithm versions and source trace

The default `adaptive` version is the data-adaptive method (`adaptive_full`) used
for the JBHI main objective results. The historical `fixed` version remains
available through `--version fixed` and `version="fixed"` in the public API.

## Source files

- `acpd/data_adaptive.py`: port of the research file
  `experiments/acpd_data_adaptive.py`, entry point `run_variant(mode="adaptive_full")`.
  Only the imports were adapted to this standalone package.
- `acpd/adaptive_v4.py`, `residual_recovery.py`, `legacy_pcg_core.py`, and
  `Detect_s1_s2_new.py`: unchanged frozen core sources. Calling the low-level
  `adaptive_v4.run_baseline` directly selects the historical fixed version.
- `acpd/pipeline.py`: public version selection and serialization of the research
  variant's temporary callback replacement.
- `acpd/denoise_wav.py`: inference-only audio wrapper. It converts stereo to mono,
  removes the mean, analyzes at 2 kHz and returns the original sample rate/length.
  A detector failure explicitly reporting `T0 estimation failed` uses the
  conservative band-pass fallback; unrelated exceptions remain visible.

Use `acpd.run_baseline(..., version="adaptive")` for the data-adaptive pipeline,
not the historical core's identically named function.

## Evidence for the default selection (2026-10-06)

The existing research runner `experiments/run_paper_tables_slim.py` dispatches
`adaptive_full` to `experiments/acpd_data_adaptive.py`. Re-aggregating its stored
three-seed runs **per original recording** yields these main-table results:

| Dataset / noise | Delta SNR | Delta SI-SDR | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| PhysioNet / synthetic | 7.47674 | 6.24744 | 0.11629 | 0.08157 |
| PASCAL / synthetic | 7.79402 | 6.68829 | 0.10792 | 0.06739 |
| PhysioNet / real recorded | 6.11186 | 4.54087 | 0.12878 | 0.08331 |
| PASCAL / real recorded | 6.57932 | 5.14541 | 0.11818 | 0.06605 |

These values round to the ACPD row in the JBHI main table. Pooling observations
directly instead gives different means because recordings have different numbers
of observations. Do not confuse observation-level and recording-level averages.

This trace establishes the default for the **main objective table**. It does not
establish that historical waveform illustrations, listening-study audio or every
ablation was generated with this same version. Those studies have their own
recording selections and provenance; do not relabel old results automatically.

## Reproduction limits

The public release includes the complete ACPD implementations, inference,
synthetic/real-noise evaluation and cumulative-module evaluation. Raw PCG/noise
datasets, patient data, original local manifests, physician rating records,
external MATLAB segmenters, third-party deep-model training and weights are not
redistributed. An identical evaluation requires identical files, record/noise
IDs, noise-manifest ordering, scenarios, seeds and preprocessing. Numerical
results may also vary with library/platform versions; no cross-platform
bit-for-bit guarantee is made.
