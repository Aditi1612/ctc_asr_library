# ctc_asr

In-process Korean ASR for the fine-tuned `Wav2Vec2ForCTC` checkpoint — **no HTTP
server, no calling the API**. Load the model once inside your app and transcribe
clips or a live mic stream directly.

This is a standalone package. It does **not** modify or import
`APIs/service_ctc/`; it only reads the checkpoint files from there by default.
The inference path (audio normalization, jamo normalization, Hangul composition,
CTC decode) is a faithful copy of the serving code, so output matches the API.

## Install

On this server (editable, from source):

```bash
conda activate ctc
pip install -e /data2/irfan/projects/APIs/ctc_asr_library
```

For other people / other machines — the code is tiny, ship it as a git repo or a
wheel:

```bash
# from a git remote
pip install "ctc-asr[hub] @ git+https://github.com/<you>/ctc_asr_library.git"

# or build a wheel and hand over the file
pip install build && python -m build          # -> dist/ctc_asr-0.1.0-*.whl
pip install "ctc_asr-0.1.0-py3-none-any.whl[hub]"
```

`torch` / `torchaudio` are heavy and hardware-specific — install the right
CUDA/CPU build first, then install this.

Optional extras: `[hub]` (HF Hub download), `[vad]` (webrtcvad — better than the
energy fallback), `[mic]` (sounddevice, for the mic demo).

## Model assets

The checkpoint (~1.2 GB) is **not** in the package. Choose a source:

### A. Hugging Face Hub (recommended for distribution)

One-time upload from this server:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
python examples/upload_to_hf.py <you>/ctc-asr-ko \
  /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/checkpoint \
  /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/vocab.json
```

That creates a **private** model repo with just the files the engine needs
(`checkpoint/config.json`, `checkpoint/model.safetensors`,
`checkpoint/preprocessor_config.json`, `vocab.json`).

Then, on any machine (with `pip install 'ctc-asr[hub]'`):

```python
from ctc_asr import Transcriber
asr = Transcriber(hf_repo="<you>/ctc-asr-ko")     # downloads + caches on first use
```

Auth for a private repo — any one of:
`huggingface-cli login`, `export HF_TOKEN=hf_xxx`, or `Transcriber(hf_repo=..., hf_token="hf_xxx")`.
Grant each teammate read access to the repo.

Env-var form (no code change): `export CTC_HF_REPO=<you>/ctc-asr-ko`.

### B. Local files

```python
Transcriber(model_dir="/abs/path/checkpoint", vocab_path="/abs/path/vocab.json")
# or
export CTC_MODEL_DIR=/abs/path/checkpoint
export CTC_VOCAB_PATH=/abs/path/vocab.json
```

### C. This server only

With no args, `Transcriber()` auto-finds the sibling
`../service_ctc/transcribe_bundle/` checkpoint. Only works where that repo lives.

**Resolution priority:** explicit `model_dir`/`vocab_path` → `hf_repo` /
`CTC_HF_REPO` → sibling `service_ctc` checkpoint.

## One-shot

```python
from ctc_asr import Transcriber

asr = Transcriber()                       # loads model once (~0.5 GB VRAM / RAM)
print(asr.transcribe("clip.wav"))         # -> "카촉틀과 토이띠차 ..."

# raw waveform (mono float32) instead of a file:
import soundfile as sf
wav, sr = sf.read("clip.wav", dtype="float32")
print(asr.transcribe(wav, input_sr=sr))

# full detail:
asr.transcribe("clip.wav", detailed=True)
# {'raw_transcription', 'transcription', 'cer', ...}
```

Accepts a file path, encoded bytes (`wav/flac/ogg`, plus `mp3/m4a` if ffmpeg is
present), or a numpy/torch waveform.

## Streaming (VAD-segmented)

Your app captures audio from anywhere and pushes mono float32 frames. Each time
the speaker pauses, the buffered utterance is transcribed and returned.

```python
asr = Transcriber()
stream = asr.stream(on_segment=lambda s: print(s.start_time, s.text))

for chunk in your_mic_source():           # any length, float32 mono
    for seg in stream.accept(chunk, input_sr=48000):   # resampled if needed
        ...                               # seg.text, seg.start_time, seg.end_time

last = stream.flush()                     # finalize trailing speech at end of stream
```

`accept()` returns a list of `Segment`s finalized during that call (usually
empty, one on a pause). It never blocks on the mic — it only runs the model when
an utterance completes.

Runnable mic demo: `python examples/stream_mic.py`

### Tuning `asr.stream(...)`

| arg | default | meaning |
|---|---|---|
| `end_silence_ms` | 600 | pause length that ends an utterance |
| `min_speech_ms` | 250 | shorter blips are dropped |
| `max_speech_ms` | 15000 | force-cut a segment that never pauses |
| `start_speech_frames` | 3 | consecutive speech frames to trigger |
| `pre_roll_ms` | 200 | audio kept before the trigger point |
| `vad_aggressiveness` | 2 | webrtcvad 0–3 (higher = stricter) |

## Notes / limits

- The model is a full-utterance CTC model — there is no token-by-token
  streaming. "Real-time" here means the transcript appears shortly after each
  pause. Latency ≈ utterance length + one short model pass (~0.1–0.6 s on GPU).
- Splitting on silence can cut a word at a boundary, so a long sentence spoken
  without pauses transcribes slightly differently than the same audio sent whole.
- One `Transcriber` is thread-safe for sequential calls but runs one inference
  at a time; use one per worker if you need parallelism.

## Layout

```
ctc_asr/
  config.py        EngineConfig + asset resolution (local / HF Hub / sibling)
  hub.py           HF Hub download + one-time checkpoint upload
  audio.py         decode / resample / normalize, processor build
  postprocess.py   jamo normalize + Hangul compose  (vendored from service_ctc)
  engine.py        Wav2Vec2CTCEngine  (mirrors service_ctc/ctc_infer.py)
  transcriber.py   Transcriber, StreamingTranscriber, Segment
  vad.py           webrtcvad wrapper + energy fallback
examples/          transcribe_file.py, stream_mic.py, upload_to_hf.py
tests/             test_smoke.py
```
