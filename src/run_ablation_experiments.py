#!/usr/bin/env python3
"""Run multimodal ablation experiments sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_multimodal.py"


EXPERIMENTS = [
    ("image_text_meta", ["--use-image", "--use-text", "--use-meta"]),
    ("image", ["--use-image", "--no-use-text", "--no-use-meta"]),
    ("text", ["--no-use-image", "--use-text", "--no-use-meta"]),
    ("meta", ["--no-use-image", "--no-use-text", "--use-meta"]),
    ("image_text", ["--use-image", "--use-text", "--no-use-meta"]),
    ("image_meta", ["--use-image", "--no-use-text", "--use-meta"]),
    ("text_meta", ["--no-use-image", "--use-text", "--use-meta"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all ablation experiments one by one.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "processed" / "multimodal_training")
    parser.add_argument("--text-column", type=str, default="title")
    parser.add_argument("--hf-endpoint", type=str, default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    for index, (name, modality_args) in enumerate(EXPERIMENTS, start=1):
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--log-every",
            str(args.log_every),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--num-workers",
            str(args.num_workers),
            "--data-dir",
            str(args.data_dir),
            "--text-column",
            args.text_column,
            "--analysis-dir",
            str(PROJECT_ROOT / "output" / "analysis" / name),
            *modality_args,
        ]

        if args.limit_train_samples is not None:
            command.extend(["--limit-train-samples", str(args.limit_train_samples)])
        if args.limit_val_samples is not None:
            command.extend(["--limit-val-samples", str(args.limit_val_samples)])

        print("=" * 80, flush=True)
        print(f"Experiment {index}/{len(EXPERIMENTS)}: {name}", flush=True)
        print("Command:", " ".join(command), flush=True)
        print("=" * 80, flush=True)

        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
