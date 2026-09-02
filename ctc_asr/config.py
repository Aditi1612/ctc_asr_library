"""Engine configuration and default asset resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Best-effort default: the checkpoint that ships with the sibling service_ctc repo
# (only meaningful on the machine that repo lives on). Off-server, use a HF Hub
# repo (hf_repo=... / CTC_HF_REPO) or explicit model_dir / CTC_MODEL_DIR.
_SIBLING_BUNDLE = Path(__file__).resolve().parents[2] / "service_ctc" / "transcribe_bundle"


def default_model_dir() -> Optional[str]:
    env = os.environ.get("CTC_MODEL_DIR")
    if env:
        return env
    candidate = _SIBLING_BUNDLE / "checkpoint"
    return str(candidate) if candidate.exists() else None


def default_vocab_path() -> Optional[str]:
    env = os.environ.get("CTC_VOCAB_PATH")
    if env:
        return env
    candidate = _SIBLING_BUNDLE / "vocab.json"
    return str(candidate) if candidate.exists() else None


@dataclass(frozen=True)
class EngineConfig:
    """Immutable inference configuration.

    Frozen so it can be used as a cache key for a loaded engine.
    """

    model_dir: str
    vocab_path: str
    target_sr: int = 16000
    min_len: int = 1600
    language: str = "korean"
    task: str = "transcribe"
    force_cpu: bool = False
    use_half: bool = False
    # Silence padding applied around each clip before inference (matches serving path).
    pad_left_seconds: float = 0.25
    pad_right_seconds: float = 0.15

    @classmethod
    def resolve(
        cls,
        model_dir: Optional[str] = None,
        vocab_path: Optional[str] = None,
        *,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "EngineConfig":
        """Resolve asset locations.

        Priority: explicit ``model_dir``/``vocab_path`` > ``hf_repo`` (arg or
        ``CTC_HF_REPO`` env) > sibling ``service_ctc`` checkpoint.
        """
        repo = hf_repo or os.environ.get("CTC_HF_REPO")

        if model_dir and vocab_path:
            md, vp = model_dir, vocab_path
        elif repo:
            # An explicit hf_repo beats the sibling fallback; a partial explicit
            # model_dir / vocab_path still overrides the matching hub path.
            from .hub import resolve_from_hub

            hub_md, hub_vp = resolve_from_hub(
                repo,
                revision=hf_revision or os.environ.get("CTC_HF_REVISION"),
                token=hf_token,
                cache_dir=hf_cache_dir or os.environ.get("CTC_HF_CACHE_DIR"),
            )
            md = model_dir or hub_md
            vp = vocab_path or hub_vp
        else:
            md = model_dir or default_model_dir()
            vp = vocab_path or default_vocab_path()

        if not md:
            raise ValueError(
                "No checkpoint found. Pass model_dir=..., set CTC_MODEL_DIR, "
                "or pass hf_repo=... / set CTC_HF_REPO."
            )
        if not vp:
            raise ValueError(
                "No vocab.json found. Pass vocab_path=..., set CTC_VOCAB_PATH, "
                "or include vocab.json in the HF repo."
            )
        return cls(model_dir=str(md), vocab_path=str(vp), **kwargs)
