# Changelog

## 0.2.0 — 2026-10-06

- Release the complete ACPD algorithm with data-adaptive `adaptive_full` as the
  default and the historical fixed-parameter method available explicitly.
- Trace the default to the JBHI objective table using recording-level
  aggregation of the existing research runs.
- Include standalone file/folder denoising, the Python API, synthetic and
  real recorded-noise evaluation, and cumulative module ablation.
- Support string file paths, mono conversion, 2 kHz analysis and restoration
  of the original input frame count/sample rate.
- Include environment definitions, version/configuration notes, generated-audio
  tests, noise/reference manifest instructions and immediate evaluation errors.
- Preserve the existing author information, artwork and Apache 2.0 license.

Validation: five public-entry regression tests passed, frozen-core file identity
and adaptive function identity checked, two synthetic mixtures through both
versions matched the original research outputs exactly, and the wheel built.
These are small-scale release checks, not a new full paper evaluation.
