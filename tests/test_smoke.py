"""Smoke tests. Require the checkpoint (sibling service_ctc repo or CTC_MODEL_DIR)
and at least one sample wav.

    pytest -q            # from the ctc_asr/ directory
"""

import glob
import os

import numpy as np
import pytest
import soundfile as sf

from ctc_asr import Transcriber
from ctc_asr.config import default_model_dir

_SAMPLE_GLOBS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "service_ctc", "samples", "*.wav"),
    os.path.join(os.path.dirname(__file__), "samples", "*.wav"),
]


def _find_sample():
    for pattern in _SAMPLE_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


pytestmark = pytest.mark.skipif(
    default_model_dir() is None, reason="no checkpoint available"
)


@pytest.fixture(scope="session")
def asr():
    return Transcriber()


def test_transcribe_file(asr):
    sample = _find_sample()
    if sample is None:
        pytest.skip("no sample wav")
    text = asr.transcribe(sample)
    assert isinstance(text, str)


def test_transcribe_waveform_matches_file(asr):
    sample = _find_sample()
    if sample is None:
        pytest.skip("no sample wav")
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    a = asr.transcribe(sample)
    b = asr.transcribe(wav, input_sr=sr)
    assert a == b


def test_streaming_segments(asr):
    sample = _find_sample()
    if sample is None:
        pytest.skip("no sample wav")
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # 300ms lead/tail silence so the VAD sees a clean utterance boundary.
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    wav = np.concatenate([pad, wav, pad])

    stream = asr.stream()
    got = []
    step = int(sr * 0.1)  # 100ms push chunks
    for i in range(0, len(wav), step):
        got.extend(stream.accept(wav[i : i + step], input_sr=sr))
    tail = stream.flush()
    if tail:
        got.append(tail)

    assert len(got) >= 1
    assert any(s.text for s in got)
