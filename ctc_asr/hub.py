"""Hugging Face Hub integration for distributing the checkpoint.

The model weights (~1.2 GB) are too large to ship inside the pip package, so
they live in a (private) HF Hub model repo. On first use the library downloads
and caches them; afterwards it loads from the local cache offline.

Expected repo layout (either works)::

    repo-root/
      config.json
      model.safetensors
      preprocessor_config.json
      vocab.json

    -- or nested --

    repo-root/
      checkpoint/
        config.json
        model.safetensors
        preprocessor_config.json
      vocab.json

Upload once with :func:`upload_checkpoint` (or the ``huggingface-cli``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# Files the engine actually needs — keeps the download small (skips optimizer.pt etc).
_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "*.safetensors",
    "pytorch_model.bin",
    "vocab.json",
    "tokenizer*.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "checkpoint/config.json",
    "checkpoint/preprocessor_config.json",
    "checkpoint/*.safetensors",
    "checkpoint/pytorch_model.bin",
]


def _require_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Hugging Face Hub support needs the 'huggingface_hub' package.\n"
            "    pip install 'ctc-asr[hub]'"
        ) from exc
    return __import__("huggingface_hub")


def _find_vocab(root: Path, model_dir: Path) -> Path:
    for candidate in (model_dir / "vocab.json", root / "vocab.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"vocab.json not found in the downloaded repo ({root}). "
        "Add it to the HF repo root or pass vocab_path=..."
    )


def resolve_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Download (or reuse cached) checkpoint; return ``(model_dir, vocab_path)``.

    ``token`` falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the
    environment, or a prior ``huggingface-cli login``.
    """
    hf = _require_hf()
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    local = hf.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        allow_patterns=_ALLOW_PATTERNS,
    )
    root = Path(local)
    model_dir = root / "checkpoint" if (root / "checkpoint").is_dir() else root
    vocab_path = _find_vocab(root, model_dir)
    return str(model_dir), str(vocab_path)


def upload_checkpoint(
    repo_id: str,
    checkpoint_dir: str,
    vocab_path: Optional[str] = None,
    *,
    private: bool = True,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Publish a local checkpoint to an HF Hub model repo (creating it if needed).

    Uploads only the files the engine needs. Returns the repo URL.
    """
    hf = _require_hf()
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = hf.HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    ckpt = Path(checkpoint_dir)
    wanted = ["config.json", "preprocessor_config.json"]
    wanted += [p.name for p in ckpt.glob("*.safetensors")]
    if not any(n.endswith(".safetensors") for n in wanted) and (ckpt / "pytorch_model.bin").is_file():
        wanted.append("pytorch_model.bin")

    for name in wanted:
        src = ckpt / name
        if src.is_file():
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"checkpoint/{name}",
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )

    vp = Path(vocab_path) if vocab_path else (ckpt / "vocab.json")
    if vp.is_file():
        api.upload_file(
            path_or_fileobj=str(vp),
            path_in_repo="vocab.json",
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )

    return f"https://huggingface.co/{repo_id}"
