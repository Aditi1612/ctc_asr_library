"""ctc_asr -- in-process Korean ASR (fine-tuned Wav2Vec2-CTC).

No HTTP server. Load the model once, transcribe clips or stream mic frames.

    from ctc_asr import Transcriber

    asr = Transcriber(model_dir=".../checkpoint", vocab_path=".../vocab.json")
    print(asr.transcribe("clip.wav"))

    stream = asr.stream(on_segment=lambda s: print(s.text))
    for chunk in mic_frames():          # float32 mono @ asr.sample_rate
        stream.accept(chunk)
    stream.flush()
"""

from .config import EngineConfig, default_model_dir, default_vocab_path
from .engine import Wav2Vec2CTCEngine, get_engine, per_sample_cer
from .hub import resolve_from_hub, upload_checkpoint
from .transcriber import Segment, StreamingTranscriber, Transcriber

__all__ = [
    "Transcriber",
    "StreamingTranscriber",
    "Segment",
    "EngineConfig",
    "Wav2Vec2CTCEngine",
    "get_engine",
    "per_sample_cer",
    "default_model_dir",
    "default_vocab_path",
    "resolve_from_hub",
    "upload_checkpoint",
]

__version__ = "0.1.0"
