"""High-level, in-process transcription API.

``Transcriber``          -- one-shot: audio in, text out.
``StreamingTranscriber`` -- push mic frames continuously; get a finalized
                            transcript per utterance (VAD-segmented).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import numpy as np

from . import audio as _audio
from .config import EngineConfig
from .engine import Wav2Vec2CTCEngine, get_engine
from .vad import VAD


class Transcriber:
    """Loads the model once and transcribes whole clips."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        vocab_path: Optional[str] = None,
        *,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        force_cpu: bool = False,
        use_half: bool = False,
        target_sr: int = 16000,
        min_len: int = 1600,
        cache: bool = True,
        **config_kwargs,
    ):
        self.config = EngineConfig.resolve(
            model_dir=model_dir,
            vocab_path=vocab_path,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            force_cpu=force_cpu,
            use_half=use_half,
            target_sr=target_sr,
            min_len=min_len,
            **config_kwargs,
        )
        self.engine: Wav2Vec2CTCEngine = (
            get_engine(self.config) if cache else Wav2Vec2CTCEngine(self.config)
        )

    @property
    def device(self) -> str:
        return self.engine.device.type

    @property
    def sample_rate(self) -> int:
        return self.config.target_sr

    def transcribe(
        self,
        audio: _audio.AudioLike,
        input_sr: Optional[int] = None,
        reference_text: Optional[str] = None,
        detailed: bool = False,
    ):
        """Return the Hangul transcript for ``audio``.

        ``audio`` may be a file path, encoded bytes, or a float32 mono waveform
        (pass ``input_sr`` for a raw waveform). With ``detailed=True`` returns
        the full result dict (raw text, CER, ...).
        """
        waveform = _audio.preprocess(
            audio,
            input_sr=input_sr,
            target_sr=self.config.target_sr,
            min_len=self.config.min_len,
            pad_left_seconds=self.config.pad_left_seconds,
            pad_right_seconds=self.config.pad_right_seconds,
        )
        result = self.engine.transcribe_waveform(waveform, reference_text=reference_text)
        return result if detailed else result["transcription"]

    def stream(self, **kwargs) -> "StreamingTranscriber":
        """Create a StreamingTranscriber backed by this model."""
        return StreamingTranscriber(self, **kwargs)


@dataclass
class Segment:
    """One finalized utterance."""

    text: str
    start_time: float
    end_time: float
    raw_text: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class StreamingTranscriber:
    """VAD-segmented streaming.

    Feed arbitrary-length mono float32 chunks to :meth:`accept`. When the
    speaker pauses (``end_silence_ms`` of non-speech), the buffered utterance is
    transcribed and returned as a :class:`Segment`. No partial hypotheses.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        frame_ms: int = 30,
        start_speech_frames: int = 3,
        end_silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_speech_ms: int = 15000,
        pre_roll_ms: int = 200,
        vad_aggressiveness: int = 2,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ):
        self.transcriber = transcriber
        self.sample_rate = transcriber.sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(self.sample_rate * frame_ms / 1000)
        self.start_speech_frames = start_speech_frames
        self.end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.max_speech_frames = max(1, round(max_speech_ms / frame_ms))
        self.pre_roll_frames = max(0, round(pre_roll_ms / frame_ms))
        self.on_segment = on_segment

        self.vad = VAD(
            sample_rate=self.sample_rate,
            frame_ms=frame_ms,
            aggressiveness=vad_aggressiveness,
        )
        self.reset()

    @property
    def vad_backend(self) -> str:
        return self.vad.backend

    def reset(self) -> None:
        self._leftover = np.zeros(0, dtype=np.float32)
        self._preroll: Deque[np.ndarray] = deque(maxlen=self.pre_roll_frames or 1)
        self._triggered = False
        self._voiced: List[np.ndarray] = []
        self._ring: Deque[bool] = deque(maxlen=self.start_speech_frames)
        self._trailing_silence = 0
        self._speech_frames_in_seg = 0
        self._n_frames = 0          # total frames consumed
        self._seg_start_frame = 0

    # ------------------------------------------------------------------ #

    def accept(self, pcm, input_sr: Optional[int] = None) -> List[Segment]:
        """Consume a chunk; return any utterances finalized during this call."""
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if input_sr is not None and input_sr != self.sample_rate:
            samples = _audio.to_waveform(
                samples, input_sr=input_sr, target_sr=self.sample_rate
            )

        buf = np.concatenate([self._leftover, samples])
        n_full = len(buf) // self.frame_len
        segments: List[Segment] = []

        for i in range(n_full):
            frame = buf[i * self.frame_len : (i + 1) * self.frame_len]
            seg = self._push_frame(frame)
            if seg is not None:
                segments.append(seg)

        self._leftover = buf[n_full * self.frame_len :].copy()
        return segments

    def flush(self) -> Optional[Segment]:
        """Force-finalize whatever speech is currently buffered (call at end of stream)."""
        seg = None
        if self._triggered:
            seg = self._finalize()
        self.reset()
        return seg

    # ------------------------------------------------------------------ #

    def _push_frame(self, frame: np.ndarray) -> Optional[Segment]:
        self._n_frames += 1
        is_speech = self.vad.is_speech(frame)

        if not self._triggered:
            self._preroll.append(frame)
            self._ring.append(is_speech)
            if self._ring.count(True) >= self.start_speech_frames:
                self._triggered = True
                self._voiced = list(self._preroll)
                self._speech_frames_in_seg = self._ring.count(True)
                self._trailing_silence = 0
                self._seg_start_frame = self._n_frames - len(self._voiced)
                self._preroll.clear()
                self._ring.clear()
            return None

        # triggered
        self._voiced.append(frame)
        if is_speech:
            self._speech_frames_in_seg += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        if self._trailing_silence >= self.end_silence_frames:
            return self._finalize()
        if len(self._voiced) >= self.max_speech_frames:
            return self._finalize()
        return None

    def _finalize(self) -> Optional[Segment]:
        voiced = self._voiced
        speech_frames = self._speech_frames_in_seg
        start_frame = self._seg_start_frame
        end_frame = start_frame + len(voiced)

        # arm for the next utterance
        self._triggered = False
        self._voiced = []
        self._ring.clear()
        self._preroll.clear()
        self._trailing_silence = 0
        self._speech_frames_in_seg = 0

        if speech_frames < self.min_speech_frames or not voiced:
            return None

        wav = np.concatenate(voiced).astype(np.float32)
        result = self.transcriber.transcribe(
            wav, input_sr=self.sample_rate, detailed=True
        )
        text = result["transcription"].strip()
        if not text:
            return None

        seg = Segment(
            text=text,
            start_time=start_frame * self.frame_ms / 1000.0,
            end_time=end_frame * self.frame_ms / 1000.0,
            raw_text=result.get("raw_transcription"),
        )
        if self.on_segment is not None:
            self.on_segment(seg)
        return seg
