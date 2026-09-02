"""Publish the local checkpoint to a private Hugging Face Hub repo (run once).

    huggingface-cli login          # or export HF_TOKEN=hf_xxx
    python examples/upload_to_hf.py <username>/ctc-asr-ko \
        /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/checkpoint \
        /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/vocab.json
"""

import sys

from ctc_asr import upload_checkpoint


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: upload_to_hf.py <repo_id> <checkpoint_dir> [vocab.json]")
    repo_id = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    vocab_path = sys.argv[3] if len(sys.argv) > 3 else None

    url = upload_checkpoint(
        repo_id=repo_id,
        checkpoint_dir=checkpoint_dir,
        vocab_path=vocab_path,
        private=True,
    )
    print("uploaded ->", url)


if __name__ == "__main__":
    main()
