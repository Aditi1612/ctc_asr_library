"""Audio loading, resampling and normalization.

Accepts the same inputs an app is likely to have on hand: a file path, raw
encoded bytes (wav/flac/ogg, plus mp3/m4a when ffmpeg is present), or an
in-memory float32 waveform.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

AudioLike = Union[str, Path, bytes, bytearray, memoryview, np.ndarray, torch.Tensor]


def create_processor(vocab_path: Union[str, Path]) -> Wav2Vec2Processor:
    """Build the Wav2Vec2 processor from the training-time vocab file."""
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        replace_word_delimiter_char="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _decode_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(audio_bytes)
    try:
        speech, sr = sf.read(buf, dtype="float32", always_2d=False)
        return np.asarray(speech, dtype=np.float32), int(sr)
    except Exception:
        # soundfile can't handle MP3/M4A — fall back to torchaudio (needs ffmpeg).
        buf.seek(0)
        waveform, sr = torchaudio.load(buf)
        return waveform.mean(dim=0).numpy().astype(np.float32), int(sr)


def to_waveform(
    audio: AudioLike,
    input_sr: int | None = None,
    target_sr: int = 16000,
) -> np.ndarray:
    """Normalize any supported input into a mono float32 numpy array at target_sr.

    ``input_sr`` is only consulted for raw ndarray/tensor input; for files and
    encoded bytes the sample rate is read from the container.
    """
    if isinstance(audio, (str, Path)):
        speech, sr = sf.read(str(audio), dtype="float32", always_2d=False)
        speech = np.asarray(speech, dtype=np.float32)
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        speech, sr = _decode_bytes(bytes(audio))
    else:
        if isinstance(audio, torch.Tensor):
            speech = audio.detach().cpu().numpy()
        else:
            speech = np.asarray(audio)
        speech = speech.astype(np.float32, copy=False)
        if input_sr is None:
            raise ValueError("input_sr is required when passing a raw waveform")
        sr = int(input_sr)

    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    if sr != target_sr:
        speech = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(speech)).float(), sr, target_sr
        ).numpy()

    return np.ascontiguousarray(speech, dtype=np.float32)


def normalize_and_pad(
    speech: np.ndarray,
    target_sr: int = 16000,
    min_len: int = 1600,
    pad_left_seconds: float = 0.25,
    pad_right_seconds: float = 0.15,
) -> torch.Tensor:
    """Mean/std normalize, add silence padding, enforce a minimum length."""
    speech = np.asarray(speech, dtype=np.float32)
    speech = speech - speech.mean()
    std = float(np.std(speech))
    if std > 0:
        speech = speech / (std + 1e-7)

    pad_left = int(pad_left_seconds * target_sr)
    pad_right = int(pad_right_seconds * target_sr)
    if pad_left or pad_right:
        speech = np.pad(speech, (pad_left, pad_right), mode="constant")

    if len(speech) < min_len:
        speech = np.pad(speech, (0, min_len - len(speech)), mode="constant")

    return torch.tensor(speech, dtype=torch.float32)


def preprocess(
    audio: AudioLike,
    input_sr: int | None = None,
    target_sr: int = 16000,
    min_len: int = 1600,
    pad_left_seconds: float = 0.25,
    pad_right_seconds: float = 0.15,
) -> torch.Tensor:
    """Full path: any input -> model-ready normalized waveform tensor."""
    speech = to_waveform(audio, input_sr=input_sr, target_sr=target_sr)
    return normalize_and_pad(
        speech,
        target_sr=target_sr,
        min_len=min_len,
        pad_left_seconds=pad_left_seconds,
        pad_right_seconds=pad_right_seconds,
    )
