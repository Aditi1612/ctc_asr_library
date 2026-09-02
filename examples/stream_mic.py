"""Live mic streaming demo (requires: pip install sounddevice).

    python examples/stream_mic.py

Speak, pause, and each utterance is transcribed when you stop talking.
Ctrl-C to quit. This file owns the microphone; the library itself only
consumes frames, so in your own app you can push audio from any source.
"""

import queue
import sys

import numpy as np

from ctc_asr import Transcriber

try:
    import sounddevice as sd
except ImportError:
    sys.exit("pip install sounddevice to run this demo")


def main() -> None:
    asr = Transcriber()
    print(f"[device={asr.device}] loading done. Speak now (Ctrl-C to stop).")

    stream = asr.stream(on_segment=lambda s: print(f"  [{s.start_time:6.2f}s] {s.text}"))
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=asr.sample_rate, channels=1, dtype="float32", callback=cb, blocksize=0
    ):
        try:
            while True:
                stream.accept(audio_q.get())
        except KeyboardInterrupt:
            final = stream.flush()
            if final:
                print(f"  [{final.start_time:6.2f}s] {final.text}")
            print("\nstopped.")


if __name__ == "__main__":
    main()
