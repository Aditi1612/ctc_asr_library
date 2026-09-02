"""Low-level Wav2Vec2-CTC inference engine.

Mirrors ``service_ctc/ctc_infer.py`` so behavior matches the API, but with no
FastAPI / web dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC

from .audio import create_processor
from .config import EngineConfig
from .postprocess import compose_hangul_from_jamo, normalize_jamo

_ENGINE_CACHE: Dict[EngineConfig, "Wav2Vec2CTCEngine"] = {}


def per_sample_cer(prediction: str, reference: str) -> float:
    """Character error rate via Levenshtein distance (no extra deps)."""
    if not reference:
        return float(prediction != reference)
    len_ref = len(reference)
    len_pred = len(prediction)
    dp = np.zeros((len_ref + 1, len_pred + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len_ref + 1)
    dp[0, :] = np.arange(len_pred + 1)
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len_ref, len_pred] / max(1, len_ref))


class Wav2Vec2CTCEngine:
    """Loads the checkpoint once; transcribes normalized waveform tensors."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.force_cpu else "cpu"
        )
        self.compute_dtype = (
            torch.float16 if (self.device.type == "cuda" and config.use_half) else torch.float32
        )
        self.processor = create_processor(Path(config.vocab_path))
        self.model = Wav2Vec2ForCTC.from_pretrained(config.model_dir)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_waveform(
        self,
        waveform: torch.Tensor,
        reference_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        speech = waveform.detach().cpu()
        if speech.ndim > 1:
            speech = speech.squeeze(0)
        if speech.ndim > 1:
            speech = speech.mean(dim=0)
        if speech.numel() < self.config.min_len:
            speech = F.pad(speech, (0, self.config.min_len - speech.numel()))

        inputs = self.processor(
            [speech.numpy()],
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device, dtype=self.compute_dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(input_values=input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw = self.processor.batch_decode(pred_ids)[0]
        normalized = normalize_jamo(raw)
        transcription = compose_hangul_from_jamo(normalized)

        reference_norm = None
        cer_value = None
        if reference_text:
            reference_norm = compose_hangul_from_jamo(normalize_jamo(reference_text))
            cer_value = per_sample_cer(transcription, reference_norm)

        return {
            "raw_transcription": raw,
            "transcription_clean": normalized,
            "transcription": transcription,
            "reference_text": reference_text,
            "reference_norm": reference_norm,
            "cer": cer_value,
        }


def get_engine(config: EngineConfig) -> Wav2Vec2CTCEngine:
    """Return a process-cached engine for this exact config."""
    engine = _ENGINE_CACHE.get(config)
    if engine is None:
        engine = Wav2Vec2CTCEngine(config)
        _ENGINE_CACHE[config] = engine
    return engine
