"""One-shot transcription of an audio file.

    python examples/transcribe_file.py /abs/path/clip.wav
"""

import sys

from ctc_asr import Transcriber


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: transcribe_file.py <audio> [model_dir] [vocab_path]")
    audio = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else None
    vocab_path = sys.argv[3] if len(sys.argv) > 3 else None

    asr = Transcriber(model_dir=model_dir, vocab_path=vocab_path)
    print(f"[device={asr.device}]")
    out = asr.transcribe(audio, detailed=True)
    print("raw :", out["raw_transcription"])
    print("text:", out["transcription"])


if __name__ == "__main__":
    main()
