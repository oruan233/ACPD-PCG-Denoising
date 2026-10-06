# Data preparation

No PCG or noise recordings are bundled. ACPD inference needs only an input WAV.
For evaluation, obtain the datasets used in the paper from their original
distributors under their own terms; the data remain separately licensed.

| Dataset | Role | Original distributor |
| --- | --- | --- |
| PhysioNet/CinC 2016 | Reference heart sounds | PhysioNet |
| PASCAL Heart Sounds Archive | Reference heart sounds | PASCAL heart-sound challenge |
| CirCor DigiScope | Cycle-source/module experiments and listening pool | PhysioNet |
| Yaseen2018/OAHS | Training data for external deep baselines | Original dataset authors |
| ICBHI 2017 | Respiratory interference | ICBHI respiratory-sound challenge |
| DEMAND | Ambient interference | Original DEMAND release |
| ARCA23K | Speech/cough/crumpling interference | Original ARCA23K release |

The release accepts your own reference WAVs and does not depend on a downloader
or hard-coded research-computer path. Public reference recordings are not
instrument-verified noise-free ground truth.

```text
data/
  reference_wavs/
    reference_001.wav
  noise/
    speech_001.wav
  noise_manifests/
    noise_manifest.csv
```

The noise manifest has `noise_id,noise_category,path,usable` columns. A reference
manifest optionally has `dataset,record_id,audio_path`. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for examples and path resolution.

Data, generated outputs, model weights and temporary files are ignored by Git.
