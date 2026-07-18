#!/usr/bin/env python3
"""One-off: restore original processor/tokenizer metadata on already-published
repos that were converted before scripts/convert_vlm.py's fix for the
processor.save_pretrained() regeneration bug (see that script's comment)."""
from __future__ import annotations

from huggingface_hub import HfApi, hf_hub_download

FILES = ["processor_config.json", "tokenizer_config.json", "special_tokens_map.json"]

REPOS = {
    "mlx-community/Ministral-3-3B-Base-2512-4bit": "mistralai/Ministral-3-3B-Base-2512",
    "mlx-community/Ministral-3-8B-Base-2512-4bit": "mistralai/Ministral-3-8B-Base-2512",
    "mlx-community/Ministral-3-14B-Base-2512-4bit": "mistralai/Ministral-3-14B-Base-2512",
}


def main() -> None:
    api = HfApi()
    for target, source in REPOS.items():
        print(f"[fix] {target} <- {source}")
        for filename in FILES:
            local = hf_hub_download(source, filename)
            api.upload_file(path_or_fileobj=local, path_in_repo=filename, repo_id=target)
            print(f"  uploaded {filename}")
        print(f"[fix] done: {target}")


if __name__ == "__main__":
    main()
