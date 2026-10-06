"""Versioned public pipeline; adaptive_full is the JBHI main-table method."""
from threading import RLock
import time

from . import adaptive_v4 as fixed_core
from .data_adaptive import run_variant

VERSIONS = ('adaptive', 'fixed')
# The original variant swaps a residual callback temporarily. Serialize public
# calls so simultaneous fixed/adaptive requests cannot mix implementations.
_PIPELINE_LOCK = RLock()


def run_baseline(clean, noisy, sr, *, version='adaptive', algorithm_input_bandpass=True,
                 segmenter='existing', gate_floor=0.3):
    """Run either version; the final waveform is result['adaptive_v3'].

    clean is only used for diagnostic metrics, never reconstruction.
    """
    if version not in VERSIONS:
        raise ValueError(f'Unknown version {version!r}; choose from {VERSIONS}')
    with _PIPELINE_LOCK:
        started = time.perf_counter()
        result = run_variant(clean, noisy, sr,
                             mode='adaptive_full' if version == 'adaptive' else 'fixed',
                             algorithm_input_bandpass=algorithm_input_bandpass,
                             segmenter=segmenter, gate_floor=gate_floor)
        result['elapsed_sec'] = time.perf_counter() - started
        result['version'] = version
        return result


compute_metrics = fixed_core.compute_metrics
