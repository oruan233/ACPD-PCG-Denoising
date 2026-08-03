<h1 align="center">Adaptive Cycle-Prior Denoising of Phonocardiograms</h1>

<p align="center">
  Boqiu Shen<sup>1</sup> &nbsp;·&nbsp; Xinxin Zhang<sup>1</sup> &nbsp;·&nbsp; Gan Pei<sup>1</sup> &nbsp;·&nbsp; Guangtao Zhai<sup>2</sup> &nbsp;·&nbsp; Menghan Hu<sup>1,*</sup>
</p>

<p align="center">
  <sup>1</sup>East China Normal University &nbsp;&nbsp;&nbsp; <sup>2</sup>Shanghai Jiao Tong University
</p>

<p align="center">
  <sup>*</sup>Corresponding author
</p>

<p align="center">
  <b>Denoising heart sounds (phonocardiograms) with an adaptive cardiac-cycle prior.</b>
</p>

<p align="center">
  For questions, please contact the corresponding author Menghan Hu (mhhu@ce.ecnu.edu.cn).
</p>

> 🚧 **Code release.** The full implementation will be released in this repository **upon publication**. The sections below describe the method, the data, and the intended usage so that the release is easy to follow once it is online.

---

## ✨ Introduction

Heart-sound (PCG) recordings collected outside quiet clinical rooms are routinely corrupted by breathing, speech, friction, and ambient noise, which overlaps the heart-sound band and often masks the low-amplitude detail (murmurs, extra sounds) that clinicians rely on. Classical filters struggle to remove such band-overlapping noise without eroding heart-sound detail, while supervised deep denoisers need clean targets and large labelled corpora and can degrade across acquisition devices.

**Adaptive Cycle-Prior Denoising (ACPD)** is a signal-processing method built on one observation: a heart sound repeats quasi-periodically with the cardiac cycle, whereas exogenous noise does not. ACPD estimates the recording's own cyclic structure and uses it as a prior to separate heart-sound-related content from dispersed noise — **selectively recovering the low-amplitude in-band detail masked under strong noise** instead of treating every non-template component as noise.

**Highlights**

- ❤️ **Cycle-consistency prior** distinguishes heart-sound residuals from dispersed noise.
- 🔬 **Detail recovery** — restores low-amplitude in-band structure (including murmur-like components) masked under strong noise.
- 🛡️ **Diagnostic-aware** — protects S1/S2-active regions and systolic/diastolic activity.
- 🔁 **Deterministic** — same input produces the same output; no learned weights or training data.

## 🧠 Method

<p align="center">
  <img src="method_overview.png" width="95%" alt="ACPD pipeline overview">
</p>

A noisy recording is resampled and band-pass filtered, then passes through three modules:

1. **Self-anchored cyclostationary skeleton** — coarse S1/S2 anchors are estimated from the noisy signal itself, and a conservative main heart-sound skeleton is reconstructed by a cross-cycle trimmed mean.
2. **Cycle-consistent residual reinjection** — the template residual is mapped into a normalized cardiac-cycle phase space, where phase energy, sign coherence, cross-cycle support, and sub-band consistency gate the backfilling of residuals that recur across cycles.
3. **Confidence-modulated output synthesis** — a Wiener-type convex combination of the candidate and the band-limited input, followed by a phase-aware gate that protects diagnostically active regions.

If reliable cardiac anchors cannot be found (e.g. severe arrhythmia, very short or very noisy input), ACPD gracefully falls back to the band-limited signal.

## 📦 Datasets

This repository ships **no data**. The method needs only your own `.wav` recordings at inference time. The paper's evaluation uses public PCG datasets (PhysioNet/CinC 2016, PASCAL Archive, CirCor DigiScope 2022, Yaseen2018/OAHS) and external noise sources (ICBHI 2017, DEMAND, ARCA23K). Please obtain each dataset from its **official source** under its own license and cite the original dataset papers; they are **not** redistributed here.

## ⚙️ Installation & 🚀 Usage

> The commands below describe how the released code will be used. They will run once the implementation is uploaded (see the code-release note above).

```bash
# 1. Clone
git clone <this-repository-url>
cd ACPD

# 2. Environment (Python 3.11; NumPy / SciPy / librosa / soundfile / PyWavelets)
conda env create -f environment.yml
conda activate acpd
```

```bash
# Denoise a single file or a folder of .wav files
python scripts/denoise.py --input noisy.wav --output denoised.wav
python scripts/denoise.py --input path/to/wavs --output path/to/out_dir
```

```python
# Python API
import soundfile as sf, acpd
noisy, sr = sf.read("noisy.wav")
denoised, info = acpd.denoise_array(noisy, sr)
```

```bash
# Controlled synthetic-noise evaluation on your own reference recordings
python scripts/evaluate.py --data-root data/reference_wavs \
    --output-dir outputs/synthetic --noise-types awgn apgn --snr-db -6 -3 0 3 6
```

ACPD needs only NumPy / SciPy / librosa / soundfile / PyWavelets — no deep-learning framework.

## 📊 Results

Denoising performance reported in the paper (recording-level means, `±` = 95% bootstrap confidence half-widths):

| Setting | ΔSNR (dB) ↑ | ΔSI-SDR (dB) ↑ | RMSE ↓ | MAE ↓ |
|---|:---:|:---:|:---:|:---:|
| Synthetic noise (AWGN + APGN, −6…6 dB) | **7.48 ± 0.07** | **6.26 ± 0.10** | **0.116 ± 0.002** | **0.081 ± 0.001** |
| Real recorded noise (ICBHI / DEMAND / ARCA23K) | **6.12 ± 0.07** | **4.56 ± 0.10** | **0.129 ± 0.002** | **0.083 ± 0.001** |

Full per-method comparison against signal-processing and deep baselines, with confidence intervals and ablations, is in the paper.

<p align="center">
  <img src="dsnr_curve.png" width="70%" alt="ΔSNR vs input SNR across methods"><br>
  <em>ΔSNR versus input SNR across methods (synthetic noise).</em>
</p>

<p align="center">
  <img src="listening_experiment.png" width="85%" alt="Physician subjective listening assessment"><br>
  <em>Physician subjective listening assessment under real recorded-noise mixtures.</em>
</p>

## 📝 Citation

If you find this work useful, please cite the paper (details will be finalized upon publication):

```bibtex
@article{shen2026acpd,
  title   = {Adaptive Cycle-Prior Denoising of Phonocardiograms},
  author  = {Shen, Boqiu and Zhang, Xinxin and Pei, Gan and Zhai, Guangtao and Hu, Menghan},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## 📄 License

License to be confirmed upon release. The public datasets referenced above remain under their own licenses and are **not** redistributed here.
