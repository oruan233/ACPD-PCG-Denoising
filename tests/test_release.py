"""Public-entry regression tests using generated audio only."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_smoke import make_synthetic_pcg
import acpd


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / '.tmp').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / '.tmp')
        cls.folder = Path(cls.temp.name)
        cls.references = cls.folder / '中文参考'
        cls.references.mkdir()
        cls.clean = make_synthetic_pcg()
        sf.write(cls.references / '心音.wav', cls.clean, 2000, subtype='FLOAT')
        rng = np.random.default_rng(8)
        noise = cls.folder / 'noise.wav'
        sf.write(noise, rng.normal(0, 0.2, 20000), 2000, subtype='FLOAT')
        cls.manifest = cls.folder / 'noise.csv'
        with cls.manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['noise_id', 'noise_category', 'path', 'usable'])
            writer.writeheader()
            writer.writerow(dict(noise_id='generated', noise_category='speech', path=noise.name, usable='1'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def command(self, *args):
        return subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                              encoding='utf-8', capture_output=True, check=True,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

    def test_silent_input_returns_silence(self):
        output, info = acpd.denoise_array(np.zeros(16000), 2000)
        self.assertTrue(np.array_equal(output, np.zeros(16000)))
        self.assertEqual(info['fallback'], 'bandpass_no_cycle')

    def test_default_and_fixed_versions(self):
        noisy = self.clean + np.random.default_rng(4).normal(0, 0.15, self.clean.size)
        for version, expected in [('adaptive', 'adaptive_full'), ('fixed', 'fixed')]:
            result = acpd.run_baseline(self.clean, noisy, 2000, version=version)
            self.assertEqual(result['parameter_mode'], expected)
            self.assertEqual(result['adaptive_v3'].shape, noisy.shape)
            self.assertTrue(np.all(np.isfinite(result['adaptive_v3'])))

    def test_file_api_accepts_strings(self):
        destination = self.folder / '中文输出' / '结果.wav'
        result = acpd.denoise_file(str(self.references / '心音.wav'), str(destination))
        self.assertTrue(destination.exists())
        self.assertEqual(result['info']['version'], 'adaptive')
        self.assertEqual(sf.info(destination).frames, self.clean.size)

    def test_synthetic_and_real_evaluation(self):
        for name, arguments in [('synthetic', ['--noise-types', 'awgn', '--snr-db', '0']),
                                ('real', ['--noise-manifest', str(self.manifest),
                                          '--noise-categories', 'speech', '--snr-db', '0'])]:
            out = self.folder / name
            self.command('scripts/evaluate.py', '--data-root', str(self.references),
                         '--output-dir', str(out), *arguments)
            with (out / 'per_record_metrics.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertTrue(np.isfinite(float(rows[0]['snr_improvement_db'])))
            self.assertEqual(rows[0]['version'], 'adaptive')
            self.assertTrue((out / 'run_summary.json').exists())

    def test_ablation_dispatcher(self):
        out = self.folder / 'ablation'
        self.command('scripts/reproduce_results.py', '--experiment', 'ablation',
                     '--data-root', str(self.references), '--output-dir', str(out), '--limit', '1')
        with (out / 'per_record_metrics.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row['variant'] for row in rows},
                         {'template_only', 'residual_reinjection', 'adaptive_fusion', 'full'})
        with (out / 'run_summary.json').open(encoding='utf-8') as stream:
            self.assertEqual(json.load(stream)['failures'], 0)


if __name__ == '__main__':
    unittest.main()
